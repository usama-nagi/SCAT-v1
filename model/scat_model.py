# model/scat_model.py
import torch
import torch.nn as nn
from model.metadata_encoder import MetadataEncoder # type: ignore
from model.e3nn_encoder import E3NNGeometryEncoder # type: ignore
from model.egnn_encoder import SE3GeometryEncoder # type: ignore
from typing import Union, Tuple, Optional


class DualChainHead(nn.Module):
    """
    Separate linear projection heads for heavy and light chains.
    During training, the appropriate head is selected for each position by `chain_mask` and the other head is skipped entirely.  This allows each head
    to learn chain-specific sequence patterns without interference from the other chain's positions, which are often very different
    """
 
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.heavy_head = nn.Linear(d_model, vocab_size)
        self.light_head = nn.Linear(d_model, vocab_size)
 
        # Initialize light head identically to heavy so the two start close
        # and diverge gradually - avoids a sudden loss spike at checkpoint
        # reload on an existing training run.
        with torch.no_grad():
            self.light_head.weight.copy_(self.heavy_head.weight)
            self.light_head.bias.copy_(self.heavy_head.bias)
 
    def forward(
        self,
        hidden: torch.Tensor,
        chain_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        hidden:     (B, L, d_model)
        chain_mask: (B, L) int64  0=heavy position, 1=light position
                    OR None  -> heavy head for all positions
 
        Returns: (B, L, vocab_size)
        """
        h_logits = self.heavy_head(hidden)
    
        if chain_mask is None:
            return h_logits
        
        l_logits = self.light_head(hidden)
        flag = chain_mask.unsqueeze(-1).float().to(hidden.device)
        return h_logits * (1.0 - flag) + l_logits * flag


class CDR3LengthEmbedding(nn.Module):
    """
    Learnable integer embedding for CDR-H3 length.
 
    Maps each length in [0, MAX_CDR3_LEN] to a d_model-dimensional vector.
 
    WHY learnable (not sinusoidal):
      - CDR3 lengths follow a discrete, non-uniform distribution (peaks at
        12-15 for human heavy chains).  A learnable embedding lets the model
        encode the structural meaning of each specific length rather than
        treating them as equally-spaced points on a line.
      - Sinusoidal embeddings assume smooth interpolation which is wrong here:
        CDR3=12 and CDR3=13 have meaningfully different loop topologies.
    """
 
    MAX_CDR3_LEN: int = 40  # Covers >99.9% of human CDR-H3 lengths (IMGT/Observed_Sequences/IMGT_human_observed_ig_aa.tab)
 
    def __init__(self, d_model: int):
        super().__init__()
        # +1 so index 0 is reserved for "unknown / no CDR3"
        self.embed = nn.Embedding(self.MAX_CDR3_LEN + 1, d_model)
        # Initialize near-zero so it has no effect at step 0 and grows
        # gradually - prevents a sudden spike in the loss when first added.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.01)
 
    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths: (B,) int64 - CDR3 length per sample.
                 0 means 'unknown'.
        Returns: (B, d_model)
        """
        return self.embed(lengths.clamp(0, self.MAX_CDR3_LEN))

def _cache_valid(cache: "torch.Tensor | None", ref: torch.Tensor) -> bool:
    """
    Check whether the zero-geometry cache can be reused for `ref`.
    Validates shape, device, AND dtype - any mismatch triggers a rebuild.
    Using str(device) sidesteps the torch.device('cuda') vs
    torch.device('cuda:0') inequality that breaks naive == comparison.
    """
    if cache is None:
        return False
    return (
        cache.shape  == ref.shape
        and str(cache.device) == str(ref.device)
        and cache.dtype  == ref.dtype
    )

class SequenceEncoderAdapter(nn.Module):
    """
    Lightweight adapter on top of the shared sequence encoder.
    Only updated on SAbDab batches (antigen_coords present).
    Decouples OAS/PDB gradient interference from the antigen-conditioning pathway.

    Identity-initialized (up.weight=0, up.bias=0) so v2 starts identically to v1
    at step 0 - no loss spike on first batch.

    v1 root cause: oas_hl seq_eff_rank collapsed 36→12 from shared gradient
    conflict between OAS AR and SAbDab NLL-ranking gradients.
    """
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, d_model)
        # Identity init: adapter has zero effect at step 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, is_sabdab: bool = False) -> torch.Tensor:
        """
        x:         (B, L, d_model)
        is_sabdab: True only when antigen_coords is present (SAbDab batches)
        """
        if not is_sabdab:
            return x
        return x + self.up(self.act(self.down(x)))

def _xattn_grad_hook(g):
    return torch.nan_to_num(g, nan=0.0).clamp(-1.0, 1.0) if g is not None else g

# Main SCAT Model
class SCATModel(nn.Module):
    """
    Structure-Conditioned Autoregressive Transformer for Antibody Design.
    
    Architecture:
    1. Geometry Encoder (SE(3)-equivariant: e3nn or egnn
    2. Sequence Context Encoder (Transformer)
    3. Structural Conditioning Stack (Cross-attention fusion)
    4. Autoregressive Decoder (Transformer)
    5. Multiple prediction heads (AR, MLM, Geometry, Interface, Affinity-future finetune)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        
        self.vocab_size = mcfg["vocab_size"]
        self.d_model = mcfg["d_model"]
        self.use_gradient_checkpointing = mcfg.get("use_gradient_checkpointing", False)
        
        # Embeddings
        self.token_emb = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=0,
        )
        self.pos_emb = nn.Embedding(
            mcfg["sequence_encoder"]["max_len"],
            self.d_model,
        )
        self._fw_drop_p = mcfg.get("fw_drop_p", 0.0)
        self.cdr_emb = nn.Embedding(2, self.d_model) # CDR vs non-CDR token embedding
        self.cdr_emb_drop_p = mcfg.get("cdr_emb_drop_p", 0.05) # Forces the model to rely on structural conditioning, not just CDR position hints.
        self.cdr3_len_emb = CDR3LengthEmbedding(self.d_model)
        self.cdr3_pos_emb = nn.Embedding(CDR3LengthEmbedding.MAX_CDR3_LEN + 1, self.d_model)
        nn.init.normal_(self.cdr3_pos_emb.weight, std=0.01)
        self.cdr3_pos_gate = nn.Parameter(torch.tensor(-2.0))  # closed at start
        
        
        # Sequence Context Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=mcfg["nhead"],
            dim_feedforward=mcfg["sequence_encoder"]["dim_feedforward"],
            dropout=mcfg["dropout"],
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=mcfg["sequence_encoder"]["num_layers"],
        )
        
        # Gradient-isolated adapter - only trained on SAbDab batches
        _adapter_cfg = mcfg.get("adapter", {})
        self.adapter_enabled = _adapter_cfg.get("enabled", False)
        if self.adapter_enabled:
            self.seq_adapter = SequenceEncoderAdapter(
                d_model=self.d_model,
                bottleneck=_adapter_cfg.get("bottleneck", 64),
            )
        else:
            self.seq_adapter = None

        # Metadata Encoder        
        self.use_metadata = mcfg.get("use_metadata", False)
        if self.use_metadata:
            self.metadata_encoder = MetadataEncoder(d_model=self.d_model)
        
        # Geometry Encoder
        self.geom_enabled = mcfg["geometry_encoder"]["enabled"]
        self._zero_geom_cache: torch.Tensor | None = None
        # Learned token substituted for combined_geom when no structure is
        # available (OAS batches). Keeps the conditioning_stack
        # query non-zero so its weights don't collapse toward zero-output
        # behavior on geometry-free batches.
        self.null_geom_token = nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.null_geom_token, std=0.02)
        
        if self.geom_enabled:
            backend = mcfg["geometry_encoder"].get("backend", "egnn")
            self.geom_fusion_proj = nn.Linear(self.d_model * 2, self.d_model)
            
            EncoderCls = E3NNGeometryEncoder if backend == "e3nn" else SE3GeometryEncoder
            enc_kwargs: dict = dict(
                d_model   = self.d_model,
                hidden_dim= mcfg["geometry_encoder"]["hidden_dim"],
                num_layers= mcfg["geometry_encoder"]["num_layers"],
                edge_cutoff= mcfg["geometry_encoder"].get("edge_cutoff", 10.0),
                max_neighbors= mcfg["geometry_encoder"].get("max_neighbors", 16),
            )
            if backend == "e3nn":
                enc_kwargs["max_ell"] = mcfg["geometry_encoder"].get("max_ell", 2)
                print("[SCATModel] Using e3nn geometry encoder")
            else:
                print("[SCATModel] Using EGNN geometry encoder")
 
            self.geom_encoder    = EncoderCls(**enc_kwargs)
            self.antigen_encoder = EncoderCls(**enc_kwargs)
            
            self.antigen_cross_attn = nn.MultiheadAttention(
                self.d_model, mcfg["nhead"],
                dropout=mcfg["dropout"], batch_first=True,
            )
            for p in self.antigen_cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.25)
            self.antigen_xattn_norm = nn.LayerNorm(self.d_model)
            self.antigen_residue_norm = nn.LayerNorm(self.d_model)
            self.antigen_xattn_gate = nn.Parameter(torch.tensor(-3.0))
            for p in self.antigen_cross_attn.parameters():
                if p.requires_grad:
                    p.register_hook(_xattn_grad_hook)
            self.antigen_xattn_out_norm = nn.LayerNorm(self.d_model)
        
        # Simplified conditioning projection 
        self.cond_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        
        # Autoregressive Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=mcfg["nhead"],
            dim_feedforward=mcfg["decoder"]["dim_feedforward"],
            dropout=mcfg["dropout"],
            batch_first=True,
        )
        # Multi-layer antigen cross-attention.
        # Layers 2 and 4 are inserted mid-decoder; layer 6 is the existing
        # antigen_cross_attn at the end of the stack.
        # All share antigen_xattn_gate (one gate to monitor/clamp).
        _xattn_layers = mcfg.get("decoder", {}).get("antigen_xattn_layers", [6])
        self.antigen_xattn_layers = _xattn_layers
        if 2 in _xattn_layers:
            self.antigen_cross_attn_2 = nn.MultiheadAttention(
                self.d_model, mcfg["nhead"],
                dropout=mcfg["dropout"], batch_first=True,
            )
            for p in self.antigen_cross_attn_2.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.25)
            self.antigen_xattn_norm_2   = nn.LayerNorm(self.d_model)
            self.antigen_xattn_out_norm_2 = nn.LayerNorm(self.d_model)
            for p in self.antigen_cross_attn_2.parameters():
                if p.requires_grad:
                    p.register_hook(_xattn_grad_hook)
            
        if 4 in _xattn_layers:
            self.antigen_cross_attn_4 = nn.MultiheadAttention(
                self.d_model, mcfg["nhead"],
                dropout=mcfg["dropout"], batch_first=True,
            )
            for p in self.antigen_cross_attn_4.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.25)
            self.antigen_xattn_norm_4   = nn.LayerNorm(self.d_model)
            self.antigen_xattn_out_norm_4 = nn.LayerNorm(self.d_model)
            for p in self.antigen_cross_attn_4.parameters():
                if p.requires_grad:
                    p.register_hook(_xattn_grad_hook)
            
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=mcfg["decoder"]["num_layers"],
        )
        
        # Prediction Heads - Language model head (AR + MLM)
        self.lm_head = DualChainHead(self.d_model, self.vocab_size)
        
        # Geometry prediction head
        if mcfg["heads"].get("geometry", False):
            self.geom_head = nn.Sequential(
                nn.Linear(self.d_model, mcfg["geometry_encoder"]["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(mcfg["geometry_encoder"]["hidden_dim"], 3)  # Predict CA coords
            )
            # Opens gradually as conditioning stack learns
            self.geom_cond_gate = nn.Parameter(torch.tensor(-1.0))
        else:
            self.geom_head = None
            self.geom_cond_gate = None
        
        # Interface prediction head
        if mcfg["heads"].get("interface", False):
            hidden = mcfg["geometry_encoder"]["hidden_dim"]
            self.interface_head = nn.Sequential(
                nn.Linear(self.d_model, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.residue_interface_head = nn.Sequential(
                nn.Linear(self.d_model, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            # Gradient clipping hooks - keeps interface heads from blowing up
            for p in self.interface_head.parameters():
                p.register_hook(lambda g: g.clamp(-1.0, 1.0) if g is not None else g)
            for p in self.residue_interface_head.parameters():
                p.register_hook(lambda g: g.clamp(-1.0, 1.0) if g is not None else g)
        else:
            self.interface_head         = None
            self.residue_interface_head = None

        # Gradient Checkpointing only for Geometry:
        if self.use_gradient_checkpointing and self.geom_enabled:
            self.geom_encoder.set_gradient_checkpointing(True) # type: ignore
            self.antigen_encoder.set_gradient_checkpointing(True) # type: ignore


    # Encoding Methods
    def _encode_sequence(self, seq_tokens, seq_mask, cdr_mask=None):
        """Encode sequence with transformer."""
        B, L = seq_tokens.shape
        device = seq_tokens.device
        
        # Positional encoding
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.token_emb(seq_tokens) + self.pos_emb(pos)

        # Always apply cdr_emb - zero mask when absent keeps cdr_emb in graph
        _cdr_in = cdr_mask.long() if cdr_mask is not None \
                  else torch.zeros(B, L, dtype=torch.long, device=device)
        cdr_signal = self.cdr_emb(_cdr_in)
        if self.cdr_emb_drop_p > 0 and self.training:
            keep = torch.bernoulli(
                torch.full((B, 1, 1), 1.0 - self.cdr_emb_drop_p, device=device)
            )
            cdr_signal = cdr_signal * keep
        x = x + cdr_signal
        
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Transformer encoding
        row_has_any = seq_mask.any(dim=1)               # [B]
        if not row_has_any.all():
            safe_mask = seq_mask.clone()
            safe_mask[~row_has_any, 0] = True           # make pos-0 visible
        else:
            safe_mask = seq_mask
 
        key_padding_mask = ~safe_mask                   # True = ignore
        enc = self.seq_encoder(x, src_key_padding_mask=key_padding_mask)
        
        enc_had_nan = not torch.isfinite(enc).all()

        if enc_had_nan:
            enc = torch.nan_to_num(enc, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pooling
        pooled = enc.masked_fill(~seq_mask.unsqueeze(-1), 0.0).sum(dim=1)
        denom = seq_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = pooled / denom
        
        return enc, pooled, enc_had_nan
    
    def _encode_geometry(
        self,
        coords,
        coord_mask,
        seq_tokens=None,
        is_antigen: bool = False,
        return_residues: bool = False,
    ) -> Union[
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        if coords is None or not self.geom_enabled:
            return (None, None, None) if return_residues else None
        encoder = self.antigen_encoder if is_antigen else self.geom_encoder
        encoder._return_residues = return_residues
        result = encoder(coords, coord_mask, seq_tokens=seq_tokens)
        encoder._return_residues = False
        # If return_residues=True but encoder returned a plain tensor
        # (e.g. degenerate early-return path missed), normalize to 3-tuple
        if return_residues and not isinstance(result, tuple):
            return result, None, None
        return result
    
    def _masked_weighted_pool(self, enc, seq_mask, cdr_mask=None):
        enc = enc.masked_fill(~seq_mask.unsqueeze(-1), 0.0)

        if cdr_mask is not None:
            weights = 1.0 + cdr_mask.float() * 2.0  # boost CDRs
            weights = weights * seq_mask.float()
        else:
            weights = seq_mask.float()

        pooled = (enc * weights.unsqueeze(-1)).sum(dim=1)
        denom = weights.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
        return pooled / denom
    
    def _get_zero_geom(self, ref: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor matching ref's shape/device/dtype, cached."""
        if not _cache_valid(self._zero_geom_cache, ref):
            self._zero_geom_cache = torch.zeros_like(ref)
        return self._zero_geom_cache  # type: ignore[return-value]
    
    # Forward Pass
    def forward(
        self,
        seq_tokens=None,
        seq_tokens_orig=None,
        geom_seq_tokens=None,
        seq_mask=None,
        cdr_mask=None,
        mlm_labels=None,
        coords=None,
        coord_mask=None,
        antigen_coords=None,
        antigen_coord_mask=None,
        interface_labels=None,
        interface_mask=None,
        task_mask=None,
        metadata=None,
        metadata_indices=None,
        chain_mask=None,
        cdr3_lengths=None,
        cdr3_mask=None,
        antigen_seq_tokens=None,
    ):
        """
        Forward pass with multi-task support.
        
        Returns dict with:
            - logits_ar: [B, L-1, V] autoregressive logits
            - logits_mlm: [B, L, V] masked LM logits
            - labels_mlm: [B, L] MLM labels
            - pred_coords: [B, L, 3] predicted coordinates
            - interface_logits: [B] or [B, L ,1] interface prediction
            - affinity_logits: [B] affinity prediction
        """
        outputs = {}
        task_mask = task_mask or {}
        
        # Sequence Encoding
        # mask CDR tokens before building conditioning context for AR task
        MASK_IDX = 4  # vocab MASK token
        if task_mask.get("AR", False) and cdr_mask is not None:
            enc_tokens_for_cond = seq_tokens.clone().masked_fill(cdr_mask.bool(), MASK_IDX) # type: ignore
            seq_enc, _, enc_had_nan = self._encode_sequence(enc_tokens_for_cond, seq_mask, None)
            seq_pooled = self._masked_weighted_pool(seq_enc, seq_mask, None) # CDR embedding still applied to full tgt during AR decoding, not encoding
        else:
            seq_enc, _, enc_had_nan = self._encode_sequence(seq_tokens, seq_mask, cdr_mask)
            seq_pooled = self._masked_weighted_pool(seq_enc, seq_mask, cdr_mask)

        if enc_had_nan:
            outputs["_seq_enc_nan"] = True
        
        # Apply SAbDab-specific adapter (identity on all other datasets)
        is_sabdab = (antigen_coords is not None)
        if self.seq_adapter is not None:
            seq_enc = self.seq_adapter(seq_enc, is_sabdab=is_sabdab)
            # Re-pool after adapter
            seq_pooled = self._masked_weighted_pool(
                seq_enc, seq_mask,
                cdr_mask if not task_mask.get("AR", False) else None
            )
        
        # Geometry Encoding        
        geom_pooled = None
        antigen_pooled = None
        antigen_residues = None
        antigen_residue_mask = None
        
        if self.geom_enabled:
            with torch.amp.autocast('cuda', enabled=False):   # type: ignore
                if coords is not None:
                    c = coords.float() if coords.dtype == torch.float16 else coords
                    MAX_GEO_LEN = 300
                    if c.shape[1] > MAX_GEO_LEN:
                        c          = c[:, :MAX_GEO_LEN]
                        coord_mask = coord_mask[:, :MAX_GEO_LEN] if coord_mask is not None else None

                    # Build the CDR mask aligned to the (possibly clipped) coord length.
                    # cdr_mask is None for PDB batches -> no masking, full backbone used.
                    _cdr_coords = None
                    if cdr_mask is not None:
                        _cdr_coords = cdr_mask[:, :c.shape[1]].bool()   # [B, Lc]

                    # mask CDR chemistry (residue type + physchem) from the
                    # antibody geom encoder. PAD(0) -> zeroed residue_emb + neutral physchem.
                    # _geom_seq_in = seq_tokens_orig if seq_tokens_orig is not None else seq_tokens
                    _geom_seq_in = (geom_seq_tokens if geom_seq_tokens is not None
                                    else seq_tokens_orig if seq_tokens_orig is not None
                                    else seq_tokens)
                    if _cdr_coords is not None and _geom_seq_in is not None:
                        _geom_seq_in = _geom_seq_in.clone()
                        _cdr_seq = cdr_mask[:, :_geom_seq_in.shape[1]].bool() # type: ignore
                        _geom_seq_in = _geom_seq_in.masked_fill(_cdr_seq, 0)

                    # zero CDR backbone coordinates and drop those residues
                    # from the graph (coord_mask=False).
                    if _cdr_coords is not None:
                        c = c.clone()
                        c = c.masked_fill(
                            _cdr_coords.unsqueeze(-1).unsqueeze(-1), 0.0
                        )
                        if coord_mask is not None:
                            coord_mask = coord_mask.clone()
                            _cdr_cm = _cdr_coords.unsqueeze(-1)            # [B, Lc, 1]
                            coord_mask[:, :c.shape[1]] = coord_mask[:, :c.shape[1]].masked_fill(
                                _cdr_cm, False
                            )
                    
                    # Coordinate jitter: breaks structural memorization of
                    # training antibody-antigen pairs. CDR and PAD positions
                    # are already zeroed/masked above, so noise only lands on
                    # valid framework backbone atoms. Training only.
                    if self.training and _cdr_coords is not None:
                        _noise = torch.randn_like(c) * 0.15   # 0.15 A std
                        if coord_mask is not None:
                            _cm = coord_mask[:, :c.shape[1]]
                            # expand to c's shape [B, L, N_atoms, 3]
                            while _cm.ndim < c.ndim:
                                _cm = _cm.unsqueeze(-1)
                            _noise = _noise * _cm.float()
                        else:
                            # No mask: at minimum zero out CDR positions
                            _noise = _noise.masked_fill(
                                _cdr_coords.unsqueeze(-1).unsqueeze(-1), 0.0
                            )
                        c = c + _noise

                    geom_pooled = self._encode_geometry(
                        c, coord_mask,
                        seq_tokens=_geom_seq_in,
                        is_antigen=False,
                    )

                if antigen_coords is not None:
                    a = antigen_coords.float() if antigen_coords.dtype == torch.float16 \
                        else antigen_coords
                    antigen_pooled, antigen_residues, antigen_residue_mask = self._encode_geometry(
                        a, antigen_coord_mask,
                        seq_tokens=antigen_seq_tokens,
                        is_antigen=True,
                        return_residues=True,
                    ) # type: ignore
                else:
                    antigen_residues      = None
                    antigen_residue_mask  = None

        if antigen_residues is not None:
            with torch.amp.autocast('cuda', enabled=False): # type: ignore
                ar = antigen_residues.float()
                ar = torch.nan_to_num(ar, nan=0.0, posinf=0.0, neginf=0.0)

                if self.training:
                    mx = ar.abs().max().item()
                    if mx > 1500.0 or (mx < 0.01 and self.training):
                        print(f"[WARN] antigen_residues dead: abs_max={mx:.3e}", flush=True)
                        # Reinitialize to small noise to break the zero trap
                        ar = torch.randn_like(ar) * 0.01

                # Outer clamp catches truly pathological values; LN handles the rest.
                ar = ar.clamp(-30.0, 30.0)
                ar = self.antigen_residue_norm(ar)

                # Explicitly zero padded positions - e3nn output at pads is garbage.
                if antigen_residue_mask is not None:
                    ar = ar * antigen_residue_mask.unsqueeze(-1).float()

                antigen_residues = ar
                
        # Metadata Encoding
        if self.use_metadata:
            _meta_input = metadata_indices if metadata_indices is not None else metadata
            if _meta_input is not None:
                meta_emb = self.metadata_encoder(_meta_input)
                if meta_emb is not None:
                    seq_pooled = seq_pooled + meta_emb
            
        # geometry-sequence fusion sees the length budget.
        if cdr3_lengths is not None:
            cdr3_len_signal = self.cdr3_len_emb(
                cdr3_lengths.to(seq_pooled.device)
            )                                            # [B, d_model]
            seq_pooled = seq_pooled + cdr3_len_signal

        # Combine antibody + antigen geometry
        # Using _get_zero_geom so shape/device/dtype all match - no silent cat errors.
        if geom_pooled is not None and antigen_pooled is not None:
            combined_geom = self.geom_fusion_proj(
                torch.cat([geom_pooled, antigen_pooled], dim=-1))
        elif geom_pooled is not None:
            combined_geom = self.geom_fusion_proj(
                torch.cat([geom_pooled, self._get_zero_geom(geom_pooled)], dim=-1))
        elif antigen_pooled is not None:
            combined_geom = self.geom_fusion_proj(
                torch.cat([self._get_zero_geom(antigen_pooled), antigen_pooled], dim=-1))
        else:
            combined_geom = self.null_geom_token.unsqueeze(0).expand(seq_pooled.shape[0], -1)
        
        # Simplified conditioning: project combined_geom through cond_proj.
        # On geometry-free batches (OAS HL-Paired), fall back to seq_pooled.
        # No gate needed - projection output is used directly.
        # cond_proj includes LayerNorm, so no separate cond_norm required.
        has_geometry = (geom_pooled is not None) or (antigen_pooled is not None)
        if has_geometry:
            cond_raw = self.cond_proj(combined_geom)
            cond = (cond_raw + seq_pooled).unsqueeze(1)
        else:
            cond = seq_pooled.unsqueeze(1)

        cond = torch.nan_to_num(cond, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # expose representations for alignment loss
        outputs["cond_repr"] = cond.squeeze(1)         # [B, d_model]
        if geom_pooled is not None or antigen_pooled is not None:
            outputs["geom_repr"] = combined_geom            # [B, d_model]
        
        outputs["seq_pooled"] = seq_pooled
        if antigen_pooled is not None:
            outputs["antigen_pooled"] = antigen_pooled
        
        # Autoregressive Decoding
        if seq_tokens is not None:
            B, L = seq_tokens.shape # type: ignore
            
            # AR teacher forcing must use the clean (pre-MLM) tokens. On OAS,
            # seq_tokens carries MLM masking; seq_tokens_orig is the clean clone.
            # On non-MLM datasets seq_tokens_orig == seq_tokens, so this is a no-op.
            ar_tokens = seq_tokens_orig if seq_tokens_orig is not None else seq_tokens

            # Shift for teacher forcing
            tgt = ar_tokens[:, :-1] # type: ignore
            tgt_mask_seq = seq_mask[:, :-1] # type: ignore
            
            # CDR infill: on antigen-bearing batches, hide ALL CDRs from the decoder input.
            # Masking all CDRs closes this channel. Framework + cdr3_pos_emb + cond +
            # antigen cross-attention remain as prediction paths.
            if (self.training
                    and task_mask.get("AR", False)
                    and cdr_mask is not None
                    and antigen_coords is not None):
                cdrs_in_tgt = cdr_mask[:, :tgt.shape[1]].bool()
                tgt = tgt.masked_fill(cdrs_in_tgt, MASK_IDX)
                
            # Framework dropout: on SAbDab batches, randomly zero framework
            # positions in the decoder input to break V-gene template reliance.
            # Forces decoder to use antigen cross-attention rather than
            # recovering H3 from framework residue context.
            _fw_drop_p = getattr(self, '_fw_drop_p', 0.0)
            if (self.training
                    and _fw_drop_p > 0.0
                    and cdr_mask is not None
                    and antigen_coords is not None):
                fw_in_tgt = ~cdr_mask[:, :tgt.shape[1]].bool()  # True = framework
                keep = torch.bernoulli(
                    torch.full((tgt.shape[0], tgt.shape[1]), 1.0 - _fw_drop_p,
                               device=tgt.device)
                ).bool()
                drop_mask = fw_in_tgt & ~keep  # framework AND not kept
                tgt = tgt.masked_fill(drop_mask, MASK_IDX)
            
            row_has_any  = tgt_mask_seq.any(dim=1)
            if not row_has_any.all():
                safe_tgt_mask = tgt_mask_seq.clone()
                safe_tgt_mask[~row_has_any, 0] = True
            else:
                safe_tgt_mask = tgt_mask_seq
            
            # Position embeddings
            pos = torch.arange(tgt.shape[1], device=tgt.device).unsqueeze(0).expand(B, -1)
            tgt_emb = self.token_emb(tgt) + self.pos_emb(pos)
            
            # Within-CDR3 positional signal: tells the decoder "you are at
            # position k of the loop", complementing cdr3_len_emb's "loop is N long".
            if cdr3_mask is not None:
                m = cdr3_mask[:, :tgt.shape[1]].long()                  # 1 inside any CDR3 block
                csum = torch.cumsum(m, dim=1)
                boundary = torch.cummax(csum * (1 - m), dim=1).values   # carry from end of prev block
                cdr3_pos_idx = ((csum - boundary) * m).clamp(0, CDR3LengthEmbedding.MAX_CDR3_LEN)
                cdr3_pos_gate_v = torch.sigmoid(self.cdr3_pos_gate)
                tgt_emb = tgt_emb + cdr3_pos_gate_v * self.cdr3_pos_emb(cdr3_pos_idx)
                
            if not torch.isfinite(tgt_emb).all():
                tgt_emb = torch.nan_to_num(tgt_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(tgt.shape[1], tgt.shape[1], device=tgt.device),
                diagonal=1,).bool()
            
            # Prepend conditioning token to encoder output so decoder can
            # attend to it as a distinct addressable position.
            # cond: [B, 1, d_model]  seq_enc: [B, L, d_model]
            # memory: [B, L+1, d_model]
            cond_tok = cond if cond.dim() == 3 else cond.unsqueeze(1)
            memory = torch.cat([cond_tok, seq_enc], dim=1)

            # Extend padding mask: conditioning token is always visible
            cond_visible = torch.ones(
                seq_mask.shape[0], 1, # type: ignore
                dtype=torch.bool, device=seq_mask.device, # type: ignore
            )
            memory_pad_mask = ~torch.cat([cond_visible, seq_mask], dim=1)
            
            # Layer-by-layer decoding to insert antigen xattn mid-stack
            dec = tgt_emb
            for layer_idx, dec_layer_module in enumerate(self.decoder.layers, start=1):
                dec = dec_layer_module(
                    tgt=dec,
                    memory=memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=~safe_tgt_mask,
                    memory_key_padding_mask=memory_pad_mask,
                )
                # Insert antigen cross-attention after layers 2 and 4
                if (layer_idx in getattr(self, 'antigen_xattn_layers', [])
                        and layer_idx != max(getattr(self, 'antigen_xattn_layers', [6]))
                        and antigen_residues is not None
                        and antigen_residue_mask is not None):
                    _xattn_mod  = getattr(self, f"antigen_cross_attn_{layer_idx}", None)
                    _xnorm_mod  = getattr(self, f"antigen_xattn_norm_{layer_idx}", None)
                    _xonorm_mod = getattr(self, f"antigen_xattn_out_norm_{layer_idx}", None)
                    if _xattn_mod is not None:
                        kpm = ~antigen_residue_mask.bool()
                        per_row_masked = kpm.all(dim=1)
                        if not per_row_masked.all():
                            safe_kpm_l = kpm.clone()
                            safe_kpm_l[per_row_masked, 0] = False
                            with torch.amp.autocast('cuda', enabled=False):
                                dec_f = dec.float()
                                attn_out_l, attn_w_l = _xattn_mod(
                                    query=_xnorm_mod(dec_f),
                                    key=antigen_residues,
                                    value=antigen_residues,
                                    key_padding_mask=safe_kpm_l,
                                    need_weights=True,
                                    average_attn_weights=True,
                                )
                                # --- READ-ONLY DIAGNOSTIC: attention-weight entropy/sharpness ---
                                # attn_w_l: [B, L_q, L_k] (averaged over heads). Compute entropy
                                # of the softmax distribution over antigen-residue positions,
                                # per query position, then mean-pool over valid rows/positions.
                                # Does NOT touch dec, gradients, or any loss term.
                                if self.training:
                                    with torch.no_grad():
                                        _aw = attn_w_l.clamp_min(1e-12)
                                        _ent = -(_aw * _aw.log()).sum(dim=-1)  # [B, L_q], nats
                                        _valid_rows = (~per_row_masked).view(-1, 1).expand_as(_ent)
                                        if _valid_rows.any():
                                            _mean_ent = _ent[_valid_rows].mean()
                                            _max_ent = torch.log(
                                                torch.tensor(float(antigen_residues.shape[1]),
                                                              device=_ent.device)
                                            )
                                            outputs.setdefault("_xattn_entropy", {})[layer_idx] = (
                                                _mean_ent.detach(),
                                                _max_ent.detach(),
                                            )
                                row_keep = (~per_row_masked).view(-1, 1, 1).float()
                                attn_out_l = attn_out_l * row_keep
                                attn_out_l = torch.nan_to_num(attn_out_l, nan=0.0)
                                attn_out_l = _xonorm_mod(attn_out_l)
                                attn_out_l = attn_out_l.clamp(-3.5, 3.5)
                                gate = torch.sigmoid(
                                        self.antigen_xattn_gate.clamp(-3.0, 3.0)
                                    ) * 0.75 + 0.05
                                dec = (dec_f + gate * attn_out_l).to(dec.dtype)
            if self.decoder.norm is not None:
                dec = self.decoder.norm(dec)
            
            if not torch.isfinite(dec).all():
                dec = torch.nan_to_num(dec, nan=0.0, posinf=1.0, neginf=-1.0)

            # gated per-residue antigen cross-attention (fp32 for stability)
            if getattr(self, "antigen_cross_attn", None) is not None:
                # hard-bounded gate, cannot escape regardless of gradient pressure:
                gate = torch.sigmoid(self.antigen_xattn_gate.clamp(-3.0, 3.0)) * 0.75 + 0.05
                outputs["antigen_xattn_gate"] = gate.detach()       # (for logging)

                if antigen_residues is not None and antigen_residue_mask is not None:
                    kpm            = ~antigen_residue_mask.bool()    # True = ignore in MHA
                    per_row_masked = kpm.all(dim=1)                  # [B] - rows with NO valid keys
                    any_row_valid  = (~per_row_masked).any()

                    if any_row_valid:
                        # Force position-0 visible for fully-masked rows so softmax stays
                        # finite; we zero those rows' attention output afterward so they
                        # contribute nothing to dec.
                        safe_kpm = kpm.clone()
                        safe_kpm[per_row_masked, 0] = False

                        with torch.amp.autocast('cuda', enabled=False):
                            dec_f    = dec.float()
                            dec_norm = self.antigen_xattn_norm(dec_f)
                            attn_out, attn_w = self.antigen_cross_attn(
                                query=dec_norm,
                                key=antigen_residues,        # already fp32 + clamped + LN'd above
                                value=antigen_residues,
                                key_padding_mask=safe_kpm,
                                need_weights=True,
                                average_attn_weights=True,
                            )
                            # --- READ-ONLY DIAGNOSTIC: attention-weight entropy/sharpness ---
                            # Post-stack injection point. Same measurement as the mid-decoder
                            # sites, keyed as "post" since there's no layer_idx here.
                            if self.training:
                                with torch.no_grad():
                                    _aw = attn_w.clamp_min(1e-12)
                                    _ent = -(_aw * _aw.log()).sum(dim=-1)  # [B, L_q], nats
                                    _valid_rows = (~per_row_masked).view(-1, 1).expand_as(_ent)
                                    if _valid_rows.any():
                                        _mean_ent = _ent[_valid_rows].mean()
                                        _max_ent = torch.log(
                                            torch.tensor(float(antigen_residues.shape[1]),
                                                          device=_ent.device)
                                        )
                                        outputs.setdefault("_xattn_entropy", {})["post"] = (
                                            _mean_ent.detach(),
                                            _max_ent.detach(),
                                        )
                            # Zero attention output for rows that had no valid antigen,
                            # so they see no contribution at all.
                            row_keep = (~per_row_masked).view(-1, 1, 1).float()
                            attn_out = attn_out * row_keep
                            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
                            attn_out = self.antigen_xattn_out_norm(attn_out)   # LN instead of tanh
                            attn_out = attn_out.clamp(-3.5, 3.5)

                            dec = (dec_f + gate * attn_out).to(dec.dtype)
            
            # Shift chain_mask to match teacher-forcing targets (drop first token)
            if chain_mask is not None:
                cm_ar = chain_mask[:, :tgt.shape[1]]   # align with shifted tgt
            else:
                cm_ar = torch.zeros(B, tgt.shape[1],
                                    dtype=torch.long, device=tgt.device)
            _raw_ar = self.lm_head(dec, cm_ar).float()
            outputs["logits_ar"] = torch.nan_to_num(
                _raw_ar, nan=0.0, posinf=100.0, neginf=-100.0
            ).clamp(-100, 100)
        
        # MLM Head
        if seq_tokens is not None: #or mlm_labels is not None:
            if chain_mask is not None:
                cm_mlm = chain_mask[:, :seq_enc.shape[1]]
            else:
                cm_mlm = torch.zeros(seq_enc.shape[0], seq_enc.shape[1],
                                    dtype=torch.long, device=seq_enc.device)
            seq_enc_mlm = seq_enc if torch.isfinite(seq_enc).all() \
                  else torch.nan_to_num(seq_enc, nan=0.0, posinf=1.0, neginf=-1.0)
            outputs["logits_mlm"] = self.lm_head(seq_enc_mlm, cm_mlm).float().clamp(-30, 30)
            outputs["mlm_labels"] = mlm_labels
        
        # Geometry Prediction Head
        if self.geom_head is not None:
            # Predict coordinates from sequence encoding
            geom_gate_v = torch.sigmoid(self.geom_cond_gate)
            # cond is [B, 1, d_model] - expand to match seq_enc [B, L, d_model]
            cond_expanded = cond.squeeze(1).unsqueeze(1).expand_as(seq_enc)
            geom_input = seq_enc + geom_gate_v * cond_expanded
            outputs["pred_coords"] = self.geom_head(geom_input)
            outputs["geom_gate"] = geom_gate_v.detach()

        # Interface Prediction Head
        if self.interface_head is not None: 
            IF_GRAD_SCALE = 0.5 if (antigen_coords is not None and self.training) else 0.10
            binding_input = IF_GRAD_SCALE * cond + (1 - IF_GRAD_SCALE) * cond.detach()
            binding_input = binding_input.squeeze(1) if binding_input.size(1) == 1 \
                            else binding_input.mean(dim=1)
            outputs["binding_logits"] = self.interface_head(binding_input)
            
            
        if self.residue_interface_head is not None:
            if interface_mask is None:
                interface_mask = torch.zeros(
                    seq_enc.shape[0], seq_enc.shape[1],
                    device=seq_enc.device, dtype=torch.float32,
                )
            L = seq_enc.size(1)
            if antigen_coords is not None and self.training:
                cond_for_iface = cond
            else:
                cond_for_iface = cond.detach()

            cond_exp = (cond_for_iface.expand(-1, L, -1) if cond_for_iface.size(1) == 1
                        else cond_for_iface[:, :L])
            residue_input = seq_enc + cond_exp
            outputs["residue_interface_logits"] = self.residue_interface_head(residue_input)
        
        # Touch params that are still conditionally unused
        if self.training:
            _z = torch.zeros(1, device=seq_tokens.device)

            # metadata_encoder: touch unconditionally - gradient path breaks on
            # PDB batches even when metadata is non-None (near-zero embeddings,
            # or seq_pooled doesn't reach loss on certain task configs).
            if self.use_metadata:
                for p in self.metadata_encoder.parameters():
                    _z = _z + p.sum() * 0.0

            # cdr3_len_emb / cdr3_pos_emb / cdr3_pos_gate: touch unconditionally.
            # These only enter the graph when cdr3_lengths/cdr3_mask are present,
            # but presence varies per dataset and per sample. Unconditional touch
            # is cheaper than tracking all cases.
            for p in self.cdr3_len_emb.parameters():
                _z = _z + p.sum() * 0.0
            for p in self.cdr3_pos_emb.parameters():
                _z = _z + p.sum() * 0.0
            _z = _z + self.cdr3_pos_gate * 0.0

            if self.geom_enabled:
                if not has_geometry:
                    # OAS: cond_proj not called, all geometry modules unused
                    for mod in [
                        self.geom_encoder,
                        self.antigen_encoder,
                        self.geom_fusion_proj,
                        self.cond_proj,
                        self.antigen_cross_attn,
                        self.antigen_xattn_norm,
                        self.antigen_residue_norm,
                        self.antigen_xattn_out_norm,
                    ]:
                        for p in mod.parameters():
                            _z = _z + p.sum() * 0.0
                    _z = _z + self.antigen_xattn_gate * 0.0
                    # mid-decoder xattn also unused
                    for _attr in [
                        "antigen_cross_attn_2", "antigen_xattn_norm_2", "antigen_xattn_out_norm_2",
                        "antigen_cross_attn_4", "antigen_xattn_norm_4", "antigen_xattn_out_norm_4",
                    ]:
                        _m = getattr(self, _attr, None)
                        if _m is not None:
                            for p in _m.parameters():
                                _z = _z + p.sum() * 0.0

                else:
                    if geom_pooled is None:
                        for p in (list(self.geom_encoder.parameters()) +
                                list(self.geom_fusion_proj.parameters())):
                            _z = _z + p.sum() * 0.0
                    if antigen_pooled is None:
                        for p in self.antigen_encoder.parameters():
                            _z = _z + p.sum() * 0.0
                    if antigen_residues is None:
                        for p in self.antigen_cross_attn.parameters():
                            _z = _z + p.sum() * 0.0
                        for p in self.antigen_xattn_norm.parameters():
                            _z = _z + p.sum() * 0.0
                        for p in self.antigen_residue_norm.parameters():
                            _z = _z + p.sum() * 0.0
                        for p in self.antigen_xattn_out_norm.parameters():
                            _z = _z + p.sum() * 0.0
                        _z = _z + self.antigen_xattn_gate * 0.0
                        # v2: mid-decoder xattn also unused when no antigen
                        for _attr in [
                            "antigen_cross_attn_2", "antigen_xattn_norm_2", "antigen_xattn_out_norm_2",
                            "antigen_cross_attn_4", "antigen_xattn_norm_4", "antigen_xattn_out_norm_4",
                        ]:
                            _m = getattr(self, _attr, None)
                            if _m is not None:
                                for p in _m.parameters():
                                    _z = _z + p.sum() * 0.0

                _z = _z + self.null_geom_token.sum() * 0.0

            # seq_adapter: touch on non-SAbDab batches
            if self.seq_adapter is not None and not is_sabdab:
                for p in self.seq_adapter.parameters():
                    _z = _z + p.sum() * 0.0

            # light_head: zero gradient on all-heavy batches
            for p in self.lm_head.light_head.parameters():
                _z = _z + p.sum() * 0.0

            outputs["_unused_touch"] = _z
   
        return outputs