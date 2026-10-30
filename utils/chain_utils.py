"""
model/chain_utils.py
====================
Utility for building the per-position chain identity mask that drives
the DualChainHead.

chain_position_mask  (B, L)  int64
  0 = heavy-chain position
  1 = light-chain position

Rules by source (mirrors SOURCE_MAP in 2_convert_oas_mmap.py):
  source 0  (OAS heavy unpaired)  -> all zeros
  source 1  (OAS light unpaired)  -> all ones
  source 2  (OAS / SAbDab paired) -> zeros for [:heavy_len], ones for [heavy_len:]
  source 3+ (PDB / BindingDB)     -> all zeros (single chain; no light)

This mask is built at collation time so it is always on CPU when it enters
the collate_fn, then moved to device in _to_device().
"""

import torch
from typing import Optional, List


# SOURCE IDs (must match 2_convert_oas_mmap.py SOURCE_MAP)
SOURCE_HEAVY  = 0
SOURCE_LIGHT  = 1
SOURCE_PAIRED = 2


def build_chain_position_mask(
    seq_len:    int,
    source:     int,
    heavy_len:  Optional[int] = None,
) -> torch.Tensor:
    """
    Build a (L,) int64 chain-position mask for a single sample.

    Parameters
    ----------
    seq_len   : total padded / actual sequence length
    source    : SOURCE_HEAVY / SOURCE_LIGHT / SOURCE_PAIRED (int)
    heavy_len : number of heavy-chain residues (required when source==PAIRED)

    Returns
    -------
    torch.Tensor of shape (seq_len,) dtype=torch.long
      0 = heavy position, 1 = light position
    """
    mask = torch.zeros(seq_len, dtype=torch.long)

    if source == SOURCE_LIGHT:
        mask.fill_(1)

    elif source == SOURCE_PAIRED:
        if heavy_len is not None and 0 < heavy_len < seq_len:
            mask[heavy_len:] = 1
        # If heavy_len is missing or nonsensical, fall back to all-heavy (safe)

    # SOURCE_HEAVY, PDB, BindingDB -> remain all zeros

    return mask


def build_chain_position_mask_batch(
    seq_tokens:  torch.Tensor,
    sources:     List[int],
    heavy_lens:  Optional[List[Optional[int]]] = None,
) -> torch.Tensor:
    """
    Vectorized version - builds (B, L) mask for a full batch.

    Parameters
    ----------
    seq_tokens : (B, L)  - used only for shape
    sources    : list of int, length B
    heavy_lens : list of Optional[int], length B (None -> treated as 0)

    Returns
    -------
    torch.Tensor (B, L) dtype=torch.long on CPU
    """
    B, L = seq_tokens.shape
    batch_mask = torch.zeros(B, L, dtype=torch.long)

    for i, src in enumerate(sources):
        hl = heavy_lens[i] if heavy_lens is not None else None
        batch_mask[i] = build_chain_position_mask(L, src, hl)

    return batch_mask

# Convenience: derive heavy_len from paired OAS / SAbDab samples
def heavy_len_from_sample(sample: dict) -> Optional[int]:
    """
    Extract heavy-chain length from a preprocessed .pt sample dict.

    Tries the keys in order:
      1. sample["lengths"]["heavy"]       - OAS paired
      2. sample["heavy"]["sequence_aa"]   - SAbDab
      3. None                             - unknown / single-chain
    """
    # OAS paired
    lengths = sample.get("lengths")
    if isinstance(lengths, dict):
        hl = lengths.get("heavy")
        if hl is not None:
            return int(hl)

    # SAbDab
    heavy = sample.get("heavy")
    if isinstance(heavy, dict):
        seq = heavy.get("sequence_aa", "")
        if seq:
            return len(seq)

    return None
