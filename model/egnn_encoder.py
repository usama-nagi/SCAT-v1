# model/egnn_encoder.py

"""
SE(3) Equivariant Geometry Encoder (EGNN-based)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNNLayer(nn.Module):
    """
    Single E(n) Equivariant Graph Neural Network layer.
    Maintains equivariance to rotations and translations.
    """
    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__()
        
        # Edge model: computes edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Node model: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate model: updates positions
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, h, coords, edge_index, edge_attr=None):
        """
        h: [N, hidden_dim] node features
        coords: [N, 3] coordinates
        edge_index: [2, E] edge connectivity
        edge_attr: [E, edge_dim] edge features (optional)
        """
        row, col = edge_index
        
        # Compute relative distances
        coord_diff = coords[row] - coords[col]  # [E, 3]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # [E, 1]
        
        # Edge features
        edge_feat = torch.cat([h[row], h[col], radial], dim=1)  # [E, 2*hidden + 1]
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        
        edge_msg = self.edge_mlp(edge_feat)  # [E, hidden_dim]
        
        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(edge_msg)  # [E, 1]
        coord_diff_weighted = coord_diff * coord_weights  # [E, 3]
        
        # Aggregate coordinate updates
        coords_update = torch.zeros_like(coords)
        coords_update.index_add_(0, row, coord_diff_weighted)
        coords = coords + coords_update
        
        # Update node features (invariant)
        agg_msg = torch.zeros_like(h)
        agg_msg.index_add_(0, row, edge_msg)
        
        h_update = self.node_mlp(torch.cat([h, agg_msg], dim=1))
        h = h + h_update
        
        return h, coords


class SE3GeometryEncoder(nn.Module):
    """
    SE(3)-equivariant encoder for 3D protein structures.
    Uses EGNN layers to process coordinates while maintaining equivariance.
    """
    def __init__(self, d_model=512, hidden_dim=128, num_layers=3, edge_cutoff=10.0):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.edge_cutoff = edge_cutoff
        
        # Initial node embedding (from coordinates)
        self.node_embedding = nn.Linear(3, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection to d_model
        self.output_proj = nn.Linear(hidden_dim, d_model)
        
    def _build_edges(self, coords, mask, cutoff=10.0):
        """
        Build edge index based on distance cutoff.
        coords: [B, L, 3]
        mask: [B, L]
        Returns: edge_index [2, E], batch_idx [E]
        """
        B, L, _ = coords.shape
        device = coords.device
        
        edges = []
        batch_idx = []
        
        for b in range(B):
            valid = mask[b].nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            
            # Compute pairwise distances
            c = coords[b, valid]  # [V, 3]
            dist = torch.cdist(c, c)  # [V, V]
            
            # Create edges within cutoff
            src, dst = (dist < cutoff).nonzero(as_tuple=True)
            
            # Remove self-loops
            mask_self = src != dst
            src, dst = src[mask_self], dst[mask_self]
            
            # Map back to original indices
            src_global = valid[src] + b * L
            dst_global = valid[dst] + b * L
            
            edges.append(torch.stack([src_global, dst_global]))
            batch_idx.extend([b] * len(src))
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), []
        
        edge_index = torch.cat(edges, dim=1)
        return edge_index, batch_idx
    
    def forward(self, coords, mask=None):
        """
        coords: [B, L, N, 3] or [B, L, 3]
        mask: [B, L, N] or [B, L]
        Returns: [B, d_model]
        """
        if coords is None:
            return None
        
        B = coords.shape[0]
        
        # Handle multi-atom case (average atoms per residue)
        if coords.ndim == 4:
            L, N = coords.shape[1], coords.shape[2]
            coords = coords.mean(dim=2)  # [B, L, 3]
            if mask is not None:
                mask = mask.any(dim=2)  # [B, L]
        else:
            L = coords.shape[1]
        
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=coords.device)
        
        # Flatten batch for EGNN processing
        coords_flat = coords.view(-1, 3)  # [B*L, 3]
        mask_flat = mask.view(-1)  # [B*L]
        
        # Initial node features
        h = self.node_embedding(coords_flat)  # [B*L, hidden_dim]
        
        # Build edges
        edge_index, _ = self._build_edges(coords, mask, self.edge_cutoff)
        
        # Apply EGNN layers
        for layer in self.egnn_layers:
            h, coords_flat = layer(h, coords_flat, edge_index)
        
        # Reshape back
        h = h.view(B, L, self.hidden_dim)  # [B, L, hidden_dim]
        
        # Pool to get representation
        h_masked = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = h_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Project to d_model
        geom_repr = self.output_proj(pooled)  # [B, d_model]
        
        return geom_repr
