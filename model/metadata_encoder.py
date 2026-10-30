# model/metadata_encoder.py

import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        self.chain_type_emb = nn.Embedding(4, d_model // 4)
        self.paired_emb     = nn.Embedding(2, d_model // 4)
        self.species_emb    = nn.Embedding(8, d_model // 4)
        self.dataset_emb    = nn.Embedding(8, d_model // 4)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, metadata_indices):
        """
        Args:
            metadata_indices: dict with keys chain_idx, paired_idx,
                              species_idx, dataset_idx — all LongTensor [B]
        Returns:
            [B, d_model]
        """
        if metadata_indices is None:
            return None

        device = next(self.parameters()).device

        chain_emb   = self.chain_type_emb(metadata_indices['chain_idx'].to(device))
        paired_emb  = self.paired_emb(metadata_indices['paired_idx'].to(device))
        species_emb = self.species_emb(metadata_indices['species_idx'].to(device))
        dataset_emb = self.dataset_emb(metadata_indices['dataset_idx'].to(device))

        meta_emb = torch.cat([chain_emb, paired_emb, species_emb, dataset_emb], dim=-1)
        return self.proj(meta_emb)