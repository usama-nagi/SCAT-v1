#!/usr/bin/env python3
"""
SE(3)-equivariant geometry encoder using e3nn.

Node features are kept as scalars only (irreps `hidden_dim x 0e`).
Scalar irreps are SO(3)-*invariant* under the tensor product, which means
the encoder is fully invariant to rotations and translations rather than
equivariant in the strict sense (equivariant would require the node
features themselves to transform under rotations).  This design choice
trades representational richness for numerical stability and ease of
batch normalization.  Described as a "scalar-dominated
geometry encoder with SO(3)-invariant global pooling."

Translation invariance is enforced explicitly by mean-centring
coordinates before the node embedding (see forward()).
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear


# -------------------------
# RBF edge expansion
# -------------------------
class RBFExpansion(nn.Module):
    """
    Gaussian radial basis function expansion of scalar distances.
    Maps a scalar distance to a K-dimensional feature vector.
    """
    def __init__(self, d_min: float = 0.0, d_max: float = 10.0, num_rbf: int = 16):
        super().__init__()
        self.d_min   = d_min
        self.d_max   = d_max
        self.num_rbf = num_rbf
        centres = torch.linspace(d_min, d_max, num_rbf)      # [K]
        self.register_buffer("centres", centres)
        self.sigma = (d_max - d_min) / (num_rbf - 1)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """d: [...] -> [..., K]"""
        d = d.unsqueeze(-1)                                   # [..., 1]
        return torch.exp(-0.5 * ((d - self.centres) / self.sigma) ** 2) # type: ignore


def _dist_to_centroid_rbf(coords_c: torch.Tensor, rbf: "RBFExpansion") -> torch.Tensor:
    """
    Compute RBF-expanded distance-from-centroid for each node.
 
    coords_c: [N, 3]  mean-centred Ca coordinates (already centred in forward())
    Returns:  [N, NUM_RBF]  scalar invariant node features
 
    WHY: replacing nn.Linear(3, hidden_dim) as node init.
    Distance from centroid is SO(3)-invariant (||x_i|| is unchanged by rotation).
    It carries useful information about whether a residue is in the core or
    on the surface of the structure.
    """
    dist = coords_c.norm(dim=-1)           # [N]  rotation-invariant scalar
    return rbf(dist)                        # [N, NUM_RBF]
 
 
def _edge_sh_invariant_scalars(edge_sh: torch.Tensor, max_ell: int) -> torch.Tensor:
    """
    Extract per-l squared norms of spherical harmonics as invariant scalars.
 
    edge_sh:  [E, irreps_edge_attr.dim]  (concatenated Y_0, Y_1, Y_2 components)
    max_ell:  highest l used
 
    Returns:  [E, max_ell + 1]  invariant scalars  (one per l-order)
 
    WHY: with scalar-only node irreps (0e), the TP uses only the l=0 part
    of edge_sh.  The per-l norms are rotation-invariant scalars that carry
    angular information (l=1 encodes direction, l=2 encodes anisotropy).
    Feeding them into the edge gate lets the model learn to weight messages
    by the local geometry orientation without needing equivariant LayerNorm.
    """
    parts = []
    offset = 0
    for l in range(max_ell + 1):
        dim_l = 2 * l + 1
        comp  = edge_sh[:, offset : offset + dim_l]   # [E, 2l+1]
        # Squared norm (sum over m), then sqrt -> invariant positive scalar
        norm_l = comp.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
        parts.append(norm_l)
        offset += dim_l
    return torch.cat(parts, dim=-1)                    # [E, max_ell + 1]

# Amino acid order matching vocab indices 5-24 (A=5, C=6, D=7 … Y=24).
# Source: Kyte & Doolittle (1982) for hydrophobicity;
#         formal charge at pH 7.4 (integer approximation);
#         Rose et al. (1985) for relative solvent-accessible volume (0-1).
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
 
# Kyte-Doolittle hydrophobicity, normalized to [0, 1] (0=most hydrophilic)
_KD_RAW = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}
_kd_vals = [_KD_RAW[aa] for aa in _AA_ORDER]
_kd_min, _kd_max = min(_kd_vals), max(_kd_vals)
_HYDROPHOBICITY = torch.tensor(
    [(_KD_RAW[aa] - _kd_min) / (_kd_max - _kd_min) for aa in _AA_ORDER],
    dtype=torch.float32,
)
 
# Formal charge at pH 7.4 (R,K=+1; D,E=-1; H≈+0.1; others≈0)
_CHARGE = torch.tensor(
    [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    dtype=torch.float32,
)
 
# Relative side-chain volume (Rose 1985), normalized 0-1 (G=0, W=1)
_VOL_RAW = {
    "A": 67,  "C": 86,  "D": 91,  "E": 109, "F": 135,
    "G": 48,  "H": 118, "I": 124, "K": 135, "L": 124,
    "M": 124, "N": 96,  "P": 90,  "Q": 114, "R": 148,
    "S": 73,  "T": 93,  "V": 105, "W": 163, "Y": 141,
}
_vol_vals = [_VOL_RAW[aa] for aa in _AA_ORDER]
_vol_min, _vol_max = min(_vol_vals), max(_vol_vals)
_VOLUME = torch.tensor(
    [(_VOL_RAW[aa] - _vol_min) / (_vol_max - _vol_min) for aa in _AA_ORDER],
    dtype=torch.float32,
)
 
# Combined lookup table: shape (20, 3) — [hydrophobicity, charge, volume]
# Accessed by (vocab_idx - 5) for standard AAs; clamped for PAD/special tokens.
PHYSICOCHEMICAL_TABLE = torch.stack([_HYDROPHOBICITY, _CHARGE, _VOLUME], dim=1)
# Shape (20, 3): index 0=A(vocab5), ..., index 19=Y(vocab24)
 
N_PHYSCHEM = 3   # number of physicochemical features per residue
 
 
def get_physchem_features(
    seq_tokens: torch.Tensor,     # (N,) or (B, L)  long — vocab indices
    table: torch.Tensor,           # (20, 3) physicochemical lookup
) -> torch.Tensor:
    """
    Convert vocab token indices to physicochemical scalar features.
 
    Vocab layout: PAD=0, UNK=1, START=2, END=3, MASK=4, A=5 … Y=24
    AA index = token_id - 5, clamped to [0, 19] (special tokens → A row, ~neutral)
 
    Returns: same shape as seq_tokens but with trailing dim 3 appended.
    e.g. input (N,) → output (N, 3)
    """
    flat   = seq_tokens.view(-1)
    aa_idx = (flat - 5).clamp(min=0, max=19)        # map to [0,19]
    feats  = table.to(seq_tokens.device)[aa_idx]     # (N, 3)
    return feats.view(*seq_tokens.shape, 3)


# -------------------------
# Single E3NN conv layer
# -------------------------
class E3NNConvLayer(nn.Module):
    """
    One round of equivariant message passing.

    Messages: TP(h_j, Y(r_ij)) gated by a scalar derived from
              distance-based RBF edge features, where j ranges over
              node i's k nearest neighbors.
    Update:   LN(sum_j m_ij) + LN(W_self h_i) + 0.1 * W_res h_i
    """

    def __init__(self, irreps_in, irreps_hidden, irreps_edge_attr, hidden_dim, 
                 n_ell_feats: int = 1):
        super().__init__()
        self.irreps_in     = irreps_in
        self.irreps_hidden = irreps_hidden

        # Tensor product (core equivariant operation)
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_edge_attr,
            irreps_out=irreps_hidden,
            internal_weights=True,
            shared_weights=True,
        )

        self.n_ell_feats = n_ell_feats   # stored for forward
        gate_input_dim   = hidden_dim + n_ell_feats
        self.edge_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Layer norms
        self.agg_norm  = nn.LayerNorm(irreps_hidden.dim)
        self.self_norm = nn.LayerNorm(irreps_hidden.dim)

        # Self-interaction and output projection
        self.self_interaction = Linear(irreps_in, irreps_hidden)
        self.output_linear    = Linear(irreps_hidden, irreps_hidden)

        # Residual projection (only when dimensions differ)
        self.residual_proj = (
            Linear(irreps_in, irreps_hidden)
            if irreps_in.dim != irreps_hidden.dim
            else None
        )

    def forward(self, h, edge_sh, edge_features, edge_index, ell_feats=None):
        """
        h:             [N, irreps_in.dim]
        edge_sh:       [E, irreps_edge_attr.dim] spherical harmonics
        edge_features: [E, hidden_dim]             RBF-derived scalars
        edge_index:    [2, E]
        """
    
        # row = query node (whose k-NN this edge was built from)
        # col = neighbor node found for that query
        row, col = edge_index
        h_input = h
    
        # --- message ---
        # Gather the NEIGHBOR's feature (h[col]), not the query's own
        # feature. Each query node receives messages from its k nearest
        # spatial neighbors.
        messages = self.tp(h[col], edge_sh)
    
        gate = self.edge_gate(
            torch.cat([edge_features, ell_feats], dim=-1)
            if ell_feats is not None
            else torch.cat([
                edge_features,
                torch.zeros(
                    edge_features.shape[0],
                    self.n_ell_feats,
                    device=edge_features.device,
                    dtype=edge_features.dtype
                )
            ], dim=-1)
        )
    
        messages = messages * gate
    
        # --- aggregate ---
        h_agg = torch.zeros(
            h.shape[0],
            self.irreps_hidden.dim,
            device=h.device,
            dtype=h.dtype
        )
    
        # Accumulate messages at the QUERY node.
        h_agg.index_add_(0, row, messages)
        h_agg = self.agg_norm(h_agg)
    
        # --- self-interaction ---
        h_self = self.self_norm(self.self_interaction(h))
    
        # --- update + residual ---
        h_new = self.output_linear(h_agg + h_self)
        h_res = (
            self.residual_proj(h_input)
            if self.residual_proj is not None
            else h_input
        )
    
        return h_new + 0.1 * h_res


# -------------------------
# Full encoder
# -------------------------
class E3NNGeometryEncoder(nn.Module):
    """
    Scalar-dominated geometry encoder with SO(3)-invariant global pooling.

    Node features are kept as scalars (irreps `hidden_dim x 0e`).
    Translation invariance is enforced by mean-centring coordinates
    before the node embedding.
    The resulting global pooled vector is invariant to both rotations
    and translations.
    """

    NUM_RBF = 16   # RBF basis functions for edge distance encoding

    def __init__(
        self,
        d_model:      int   = 256,
        hidden_dim:   int   = 128,
        num_layers:   int   = 3,
        edge_cutoff:  float = 10.0,
        max_ell:      int   = 2,
        max_neighbors: int  = 32,
    ):
        super().__init__()
        self.d_model       = d_model
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.edge_cutoff   = edge_cutoff
        self.max_ell       = max_ell
        self.max_neighbors = max_neighbors

        # Scalar-only irreps
        self.irreps_in       = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_hidden   = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(max_ell)

        self.use_gradient_checkpointing = False

        self.node_rbf = RBFExpansion(d_min=0.0, d_max=50.0, num_rbf=self.NUM_RBF)

        # Step-4b: learnable residue type embedding (SO(3)-invariant scalar)
        # vocab size 25 (PAD/UNK/START/END/MASK + 20 AAs)
        RESIDUE_EMB_DIM = 16
        self.residue_emb = nn.Embedding(25, RESIDUE_EMB_DIM, padding_idx=0)
        nn.init.normal_(self.residue_emb.weight, std=0.02)
        self.residue_emb.weight.data[0].zero_()   # PAD → zero

        # Register physicochemical lookup as a buffer (not learned, no grad)
        self.register_buffer("physchem_table", PHYSICOCHEMICAL_TABLE.clone())

        # Combined node feature dim:
        #   NUM_RBF (dist-from-centroid) + RESIDUE_EMB_DIM + N_PHYSCHEM
        node_feat_dim = self.NUM_RBF + RESIDUE_EMB_DIM + N_PHYSCHEM
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge MLP: RBF(distance) -> hidden_dim scalar features
        self.rbf = RBFExpansion(d_min=0.0, d_max=edge_cutoff, num_rbf=self.NUM_RBF)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.NUM_RBF, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # pass n_ell_feats so gate MLP knows how many angular scalars
        _n_ell = max_ell + 1   # one per l-order: l=0,1,...,max_ell
        self._max_ell = max_ell
        self.conv_layers = nn.ModuleList([
            E3NNConvLayer(
                irreps_in        = self.irreps_hidden if i > 0 else self.irreps_in,
                irreps_hidden    = self.irreps_hidden,
                irreps_edge_attr = self.irreps_edge_attr,
                hidden_dim       = hidden_dim,
                n_ell_feats      = _n_ell,   # Step-4
            )
            for i in range(num_layers)
        ])

        # Output projection to d_model
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def set_gradient_checkpointing(self, value: bool):
        self.use_gradient_checkpointing = value

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_edges(self, coords: torch.Tensor, mask: torch.Tensor):
        B, L, _ = coords.shape
        device   = coords.device

        if not mask.any():
            return (torch.zeros((2,0),dtype=torch.long,device=device),
                    torch.zeros((0,3),device=device),
                    torch.zeros((0,),device=device))

        # Vectorized pairwise distances: avoid Python loop over B
        # Memory cost: B × L × L × 4 bytes
        # For B=16, L=280: 16 × 280 × 280 × 4 = ~50MB — acceptable
        # For large batches (B=128, L=280), fall back to chunked loop
        MAX_VECTORIZED_COST = 50_000_000   # ~50M elements
        if B * L * L <= MAX_VECTORIZED_COST:
            return self._build_edges_vectorized(coords, mask, B, L, device)
        else:
            return self._build_edges_loop(coords, mask, B, L, device)

    def _build_edges_vectorized(self, coords, mask, B, L, device):
        """Fully vectorized edge building — fast for small B or small L."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)        # [B, L, L, 3]
        dist = diff.norm(dim=-1)                                 # [B, L, L]

        # Mask invalid pairs and self-loops
        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)       # [B, L, L]
        dist = dist.masked_fill(~valid_pair, self.edge_cutoff + 1.0)
        dist.diagonal(dim1=1, dim2=2).fill_(self.edge_cutoff + 1.0)

        k = min(self.max_neighbors, L - 1)
        dist_topk, idx_topk = dist.topk(k, dim=2, largest=False)  # [B, L, k]

        # Keep only edges within cutoff
        keep = dist_topk < self.edge_cutoff                      # [B, L, k]

        b_idx   = torch.arange(B, device=device).view(B,1,1).expand(B,L,k)
        src_loc = torch.arange(L, device=device).view(1,L,1).expand(B,L,k)

        # Global indices
        src_g = (b_idx * L + src_loc)[keep]
        dst_g = (b_idx * L + idx_topk)[keep]

        if src_g.numel() == 0:
            return (torch.zeros((2,0),dtype=torch.long,device=device),
                    torch.zeros((0,3),device=device),
                    torch.zeros((0,),device=device))

        # Gather vectors and distances
        edge_vec  = diff[b_idx[keep], src_loc[keep], idx_topk[keep]]
        edge_len  = dist_topk[keep]

        valid_e = edge_len > 1e-6
        return (torch.stack([src_g[valid_e], dst_g[valid_e]]),
                edge_vec[valid_e],
                edge_len[valid_e])

    def _build_edges_loop(self, coords, mask, B, L, device):
        """Original per-sample loop — used when batch is too large to vectorize."""
        edges, edge_vecs, edge_lengths = [], [], []
        for b in range(B):
            valid = mask[b].nonzero(as_tuple=True)[0]
            if len(valid) < 2:
                continue
            c    = coords[b, valid]
            diff = c.unsqueeze(0) - c.unsqueeze(1)
            dist = diff.norm(dim=-1)
            k    = min(self.max_neighbors + 1, len(valid))
            dist_topk, idx_topk = dist.topk(k, dim=1, largest=False)
            V   = len(valid)
            src = torch.arange(V, device=device).unsqueeze(1).expand(-1, k)
            dst = idx_topk
            keep = (dist_topk < self.edge_cutoff) & (src != dst)
            src2, dst2, d2 = src[keep], dst[keep], dist_topk[keep]
            if src2.numel() == 0:
                continue
            sg = valid[src2] + b * L
            dg = valid[dst2] + b * L
            edges.append(torch.stack([sg, dg]))
            edge_vecs.append(diff[src2, dst2])
            edge_lengths.append(d2)
        if not edges:
            return (torch.zeros((2,0),dtype=torch.long,device=device),
                    torch.zeros((0,3),device=device),
                    torch.zeros((0,),device=device))
        ei = torch.cat(edges, 1)
        ev = torch.cat(edge_vecs, 0)
        el = torch.cat(edge_lengths, 0)
        valid_e = el > 1e-6
        return ei[:, valid_e], ev[valid_e], el[valid_e]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, coords: torch.Tensor, mask: torch.Tensor = None, # type: ignore
                seq_tokens: torch.Tensor = None): # type: ignore
        """
        coords: [B, L, N_A, 3] multi-atom  OR  [B, L, 3] Ca-only
        mask:   [B, L, N_A]    OR  [B, L]   (validity)

        Returns: [B, d_model]  SO(3)-invariant global geometry embedding,
                or None if coords is None.
        """
        if coords is None:
            return None

        B = coords.shape[0]

        # ---- 1. Extract Ca position ----
        if coords.ndim == 4:
            # shape[2] > 1: true multi-atom (N=0, CA=1, C=2, O=3) -> index 1
            # shape[2] == 1: Ca-only stored as [B, L, 1, 3] (SAbDab/PDB loaders) -> index 0
            ca_idx = 1 if coords.shape[2] > 1 else 0
            coords_ca = coords[:, :, ca_idx, :]
            if mask is not None and mask.ndim == 3:
                mask = mask[:, :, ca_idx]
        else:
            coords_ca = coords                  # [B, L, 3] already Ca

        L = coords_ca.shape[1]

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=coords_ca.device)

        # ---- 2. Mean-centre coordinates (enforces translation invariance) ----
        # Compute per-sample centroid over valid residues only
        mask_f   = mask.float().unsqueeze(-1)               # [B, L, 1]
        centroid = (coords_ca * mask_f).sum(dim=1, keepdim=True) \
                / mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)
        coords_c = coords_ca - centroid                      # [B, L, 3]

        # Zero-out invalid residue positions so they don't leak into edges
        coords_c = coords_c * mask_f

        # ---- 3. Initial node embedding ----
        coords_flat    = coords_c.view(-1, 3)                 # [B*L, 3]
 
        # Geometry: distance from centroid (rotation-invariant scalar)
        node_rbf_feats = _dist_to_centroid_rbf(coords_flat, self.node_rbf)
 
        # Chemistry: residue type embedding + physicochemical scalars
        if seq_tokens is not None:
            # Align seq_tokens to coordinate length L.
            # They can differ when sequence and structure were truncated
            # independently (e.g. smart_truncate vs coord padding).
            S = seq_tokens.shape[1]
            if S >= L:
                tok_flat = seq_tokens[:, :L].reshape(-1)       # [B*L]
            else:
                # Pad missing positions with PAD=0 → zero chemistry features
                pad      = torch.zeros(
                    seq_tokens.shape[0], L - S,
                    dtype=torch.long, device=seq_tokens.device,
                )
                tok_flat = torch.cat([seq_tokens, pad], dim=1).reshape(-1)  # [B*L]
            res_emb   = self.residue_emb(tok_flat)             # [B*L, 16]
            physchem  = get_physchem_features(tok_flat,
                            self.physchem_table)               # type: ignore # [B*L, 3]
            node_feats = torch.cat([node_rbf_feats, res_emb, physchem], dim=-1)
        else:
            # PDB pretraining batches have no seq_tokens -> use zeros for
            # chemistry channels so the geometry path is unchanged
            chem_zeros = torch.zeros(
                coords_flat.shape[0],
                self.residue_emb.embedding_dim + N_PHYSCHEM,
                device=coords_flat.device, dtype=coords_flat.dtype,
            )
            node_feats = torch.cat([node_rbf_feats, chem_zeros], dim=-1)
 
        h = self.node_embedding(node_feats)

        # ---- 4. Graph construction ----
        edge_index, edge_vec, edge_length = self._build_edges(coords_c, mask)

        if edge_index.shape[1] == 0:
            # Degenerate case: pool node features directly
            h_2d    = h.view(B, L, -1)
            h_mask  = h_2d.masked_fill(~mask.unsqueeze(-1), 0.0)
            pooled  = h_mask.sum(1) / mask.float().sum(1, keepdim=True).clamp(1)
            projected = self.output_proj(pooled)
            if getattr(self, '_return_residues', False):
                h_proj = self.output_proj(h_2d)
                return projected, h_proj, mask
            return projected
            # return self.output_proj(pooled)

        # ---- 5. Edge attributes ----
        # Spherical harmonics of unit direction vectors (equivariant)
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr,
            edge_vec,
            normalize=True,
            normalization="component",
        )                                                     # [E, irreps_edge.dim]

        # RBF-encoded scalar distance features (rotation-invariant)
        edge_features = self.edge_mlp(self.rbf(edge_length)) # [E, hidden_dim]

        # ---- 6. Message passing ----
        ell_feats = _edge_sh_invariant_scalars(edge_sh, self._max_ell)  # [E, max_ell+1]
        for layer in self.conv_layers:
            if self.use_gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint( # type: ignore
                    layer, h, edge_sh, edge_features, edge_index, ell_feats,
                    use_reentrant=False,
                )
            else:
                h = layer(h, edge_sh, edge_features, edge_index, ell_feats)

        # ---- 7. Validity-masked mean pooling ----
        h_2d   = h.view(B, L, -1)                            # [B, L, hidden_dim]
        h_mask = h_2d.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = h_mask.sum(1) / mask.float().sum(1, keepdim=True).clamp(1)

        projected = self.output_proj(pooled)                  # [B, d_model]

        if getattr(self, '_return_residues', False):
            node_mask = mask                                   # [B, L]
            h_proj    = self.output_proj(h_2d)                # [B, L, d_model]
            return projected, h_proj, node_mask
        return projected                                       # [B, d_model]