# data/collate.py
import numpy as np
import torch
from utils.mlm import apply_mlm_mask
from utils.cdr_utils import extract_cdr3_mask
from utils.chain_utils import build_chain_position_mask_batch
import time

_SOURCE_TO_INT = {"heavy": 0, "light": 1, "paired": 2}

def scat_collate_fn(batch):
    """
    SCAT collate function.

    Supports:
    - sequence-only
    - structure-only
    - mixed batches
    
    Returns properly batched tensors + task_mask dict.
    """

    # Filter unusable samples    
    usable = []
    for s in batch:
        if s is None:
            continue
        has_seq = "seq_tokens" in s and s["seq_tokens"] is not None
        has_struct = "coords" in s and s["coords"] is not None
        if has_seq or has_struct:
            usable.append(s)

    if len(usable) == 0:
        return None

    batch = usable

    # Final dynamic batch size
    B = len(batch)

    # Aggregate task masks from all samples
    task_mask = {
        "AR": any(s.get("task_mask", {}).get("AR", False) for s in batch),
        "MLM": any(s.get("task_mask", {}).get("MLM", False) for s in batch),
        "GEO": any(s.get("task_mask", {}).get("GEO", False) for s in batch),
        "IF": any(s.get("task_mask", {}).get("IF", False) for s in batch),
    }

    # Per-sample task masks
    sample_task_mask = {
        "AR": torch.tensor(
            [s.get("task_mask", {}).get("AR", False) for s in batch],
            dtype=torch.bool,
        ),
        "MLM": torch.tensor(
            [s.get("task_mask", {}).get("MLM", False) for s in batch],
            dtype=torch.bool,
        ),
        "GEO": torch.tensor(
            [s.get("task_mask", {}).get("GEO", False) for s in batch],
            dtype=torch.bool,
        ),
        "IF": torch.tensor(
            [s.get("task_mask", {}).get("IF", False) for s in batch],
            dtype=torch.bool,
        ),
    }

    # Sequence padding    
    has_seq = any("seq_tokens" in s and s["seq_tokens"] is not None for s in batch)

    if has_seq:
        seq_lens = [
            len(s["seq_tokens"]) if s.get("seq_tokens") is not None else 0
            for s in batch
        ]
        max_len = max(seq_lens)

        seq_tokens = torch.zeros(B, max_len, dtype=torch.long)
        seq_mask = torch.zeros(B, max_len, dtype=torch.bool)
        cdr_mask = torch.zeros(B, max_len, dtype=torch.bool)
        contact_mask = torch.zeros(B, max_len, dtype=torch.float32)

        for i, s in enumerate(batch):
            if s.get("seq_tokens") is None:
                continue
            L = len(s["seq_tokens"])
            seq_tokens[i, :L] = torch.as_tensor(s["seq_tokens"], dtype=torch.long)
            seq_mask[i, :L] = torch.as_tensor(
                s.get("seq_mask", torch.ones(L)), dtype=torch.bool
            )

            if s.get("cdr_mask") is not None:
                cdr_mask[i, :L] = torch.as_tensor(s["cdr_mask"], dtype=torch.bool)

            if s.get("contact_mask") is not None:
                contact_mask[i, :L] = torch.as_tensor(s["contact_mask"], dtype=torch.float32)
                
        def _resolve_source_id(s: dict) -> int:
            # Explicit source_id takes priority (OAS sets this correctly)
            if s.get("source_id") is not None:
                return int(s["source_id"])
            # SAbDab: derive from paired flag
            if s.get("paired", False):
                return 2   # SOURCE_PAIRED
            src = s.get("source", "")
            if src == "light":
                return 1
            return 0  # heavy

        def _resolve_heavy_len(s: dict) -> int | None:
            # Explicit heavy_len_actual takes priority (OAS sets this)
            if s.get("heavy_len_actual") is not None:
                return s["heavy_len_actual"]
            # SAbDab paired: derive from heavy sequence length
            heavy = s.get("heavy")
            if isinstance(heavy, dict):
                seq = heavy.get("sequence_aa", "")
                if seq:
                    return len(seq)
            return None

        sources    = [_resolve_source_id(s)  for s in batch]
        heavy_lens = [_resolve_heavy_len(s)  for s in batch]

        chain_position_mask = build_chain_position_mask_batch(
            seq_tokens,
            sources,
            heavy_lens
        )
    else:
        seq_tokens = None
        seq_mask = None
        cdr_mask = None
        contact_mask = None
        chain_position_mask = None

    # MLM masking (collate-time, universal) 
    mlm_labels = None
    cdr3_mask  = None
    cdr12_mask = None
    cdr3_lengths = None

    if cdr_mask is not None and cdr_mask.any():
        sources_t = torch.tensor(
            [
                2 if s.get("paired", False) else _SOURCE_TO_INT.get(s.get("source"), 0)
                for s in batch
            ],
            dtype=torch.long,
        )
        cdr3_mask  = extract_cdr3_mask(cdr_mask, sources_t)
        cdr12_mask = cdr_mask & ~cdr3_mask
        stored_lens = [s.get("cdr3_len_actual", 0) for s in batch]
        if any(l > 0 for l in stored_lens):
            cdr3_lengths = torch.tensor(stored_lens, dtype=torch.long)
        else:
            if chain_position_mask is not None:
                heavy_pos = (chain_position_mask == 0)
                cdr3_lengths = (cdr3_mask & heavy_pos).sum(dim=1).long()
            else:
                cdr3_lengths = cdr3_mask.sum(dim=1).long()
    
    seq_tokens_orig = seq_tokens.clone() if seq_tokens is not None else None
    if task_mask["MLM"] and seq_tokens is not None:
        
        # Dynamic masking strategy based on data type
        has_cdr_data = cdr_mask is not None and cdr_mask.sum() > 0
        
        # Check if batch contains antibody datasets
        is_antibody_batch = any(
            s.get("dataset") in ["oas", "oas_hl", "oas_paired", "sabdab"]
            for s in batch
        )
        
        # Use CDR-only masking for antibody data with CDR annotations
        if is_antibody_batch and has_cdr_data:
            # 70% CDR-focused, 30% global masking
            cdr_bias = torch.rand_like(seq_mask.float()) < 0.7 # type: ignore

            candidate_mask = torch.where(cdr_bias, cdr_mask, seq_mask) # type: ignore

        else:
            candidate_mask = seq_mask
        
        seq_tokens, mlm_labels, mlm_mask = apply_mlm_mask(
            seq_tokens=seq_tokens,
            seq_mask=seq_mask,
            candidate_mask=candidate_mask,
            mask_prob=0.15,
            cdr3_mask=cdr3_mask,
            cdr12_mask=cdr12_mask,
        )
        
        # Disable MLM if no positions were masked
        if (mlm_labels != -100).sum() == 0:
            task_mask["MLM"] = False
            mlm_labels = None

    if mlm_labels is not None:
        assert mlm_labels.shape == seq_tokens.shape # type: ignore

    # Structure padding    
    has_struct = any(s.get("coords") is not None for s in batch)

    if has_struct:
        max_l = 0
        max_n = 1

        for s in batch:
            c = s.get("coords")
            if c is None:
                continue
            c = torch.as_tensor(c)
            max_l = max(max_l, c.shape[0])
            if c.ndim == 3:
                max_n = max(max_n, c.shape[1])

        coords = torch.zeros(B, max_l, max_n, 3)
        coord_mask = torch.zeros(B, max_l, max_n, dtype=torch.bool)

        for i, s in enumerate(batch):
            c = s.get("coords")
            if c is None:
                continue

            c = torch.as_tensor(c, dtype=torch.float)
            if c.ndim == 2:
                c = c.unsqueeze(1)

            L, N = c.shape[0], c.shape[1]
            coords[i, :L, :N] = c

            m = s.get("coord_mask")
            if m is None:
                coord_mask[i, :L, :N] = True
            else:
                m = torch.as_tensor(m, dtype=torch.bool)
                if m.ndim == 1:
                    m = m.unsqueeze(-1)
                coord_mask[i, :L, :N] = m
    else:
        coords = None
        coord_mask = None

    # Antigen conditioning    
    has_antigen = any(s.get("antigen_coords") is not None for s in batch)

    if has_antigen:
        max_m = 0
        max_n = 1

        for s in batch:
            a = s.get("antigen_coords")
            if a is None:
                continue
            a = torch.as_tensor(a)
            max_m = max(max_m, a.shape[0])
            if a.ndim == 3:
                max_n = max(max_n, a.shape[1])

        antigen_coords = torch.zeros(B, max_m, max_n, 3)
        antigen_mask = torch.zeros(B, max_m, max_n, dtype=torch.bool)

        for i, s in enumerate(batch):
            a = s.get("antigen_coords")
            if a is None:
                continue

            a = torch.as_tensor(a, dtype=torch.float)
            if a.ndim == 2:
                a = a.unsqueeze(1)

            M, N = a.shape[0], a.shape[1]
            antigen_coords[i, :M, :N] = a

            am = s.get("antigen_coord_mask")
            if am is None:
                antigen_mask[i, :M, :N] = True
            else:
                am = torch.as_tensor(am, dtype=torch.bool)
                if am.ndim == 1:
                    am = am.unsqueeze(-1)
                antigen_mask[i, :M, :N] = am
    else:
        antigen_coords = None
        antigen_mask = None
        
    # antigen residue tokens for chemistry conditioning
    antigen_seq_tok_list = [s.get("antigen_seq_tokens") for s in batch]
    has_ag_seq = any(t is not None for t in antigen_seq_tok_list)
    if has_ag_seq:
        max_ag_seq = max(
            (len(t) for t in antigen_seq_tok_list if t is not None), default=0
        )
        padded_ag_seq = []
        for t in antigen_seq_tok_list:
            if t is None:
                padded_ag_seq.append(torch.zeros(max_ag_seq, dtype=torch.long))
            else:
                pad = max_ag_seq - len(t)
                padded_ag_seq.append(
                    torch.cat([t, torch.zeros(pad, dtype=torch.long)])
                    if pad > 0 else t
                )
        antigen_seq_tokens = torch.stack(padded_ag_seq)   # (B, Lag)
    else:
        antigen_seq_tokens = None

    # Interface labels & masks (for IF task)    
    interface_labels = None
    interface_mask = None
    
    if task_mask["IF"]:
        # 1. Sequence-level binding labels (SAbDab)
        labels = []
        for s in batch:
            label = s.get("interface_labels")
            if label is not None:
                if isinstance(label, torch.Tensor):
                    labels.append(label.item())
                else:
                    labels.append(float(label))
            else:
                labels.append(float("nan"))
        
        if labels:
            interface_labels = torch.tensor(labels, dtype=torch.float)
        
        # 2. Residue-level interface masks (SAbDab with structure)
        has_residue_masks = any(
                s.get("has_residue_interface", False) and 
                s.get("interface_mask") is not None
                for s in batch
            )
        
        if has_residue_masks:
            # Get max sequence length for padding
            if seq_tokens is not None:
                max_len = seq_tokens.shape[1]
            else:
                # Fallback: find max from interface masks themselves
                max_len = max(
                    (len(s.get("interface_mask")) 
                     if s.get("interface_mask") is not None 
                     else 0)
                    for s in batch
                )
            
            if max_len > 0:
                interface_mask = torch.full((B, max_len), float("nan"))
                
                for i, s in enumerate(batch):
                    if not s.get("has_residue_interface", False):
                        continue
                        
                    mask = s.get("interface_mask")
                    if mask is not None:
                        # Convert to tensor if needed
                        if isinstance(mask, torch.Tensor):
                            mask_tensor = mask
                        elif isinstance(mask, (list, tuple, np.ndarray)):
                            mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                        else:
                            continue
                        
                        L = min(len(mask_tensor), max_len)
                        interface_mask[i, :L] = mask_tensor[:L]

    # Metadata
    metadata = []

    # Pre-compute index tensors for MetadataEncoder (avoids Python loop in forward)
    _chain_vocab   = {'heavy': 0, 'light': 1, 'single': 2, 'unknown': 3}
    _species_vocab = {'human': 0, 'mouse': 1, 'rat': 2, 'unknown': 7}
    _dataset_vocab = {'oas': 0, 'sabdab': 1, 'unknown': 7}

    _chain_ids   = []
    _paired_ids  = []
    _species_ids = []
    _dataset_ids = []

    for s in batch:
        meta_source = (s.get('source') or s.get('dataset') or "").lower()
        meta_dict = {
            'chain':  s.get('chain'),
            'paired': s.get('paired', False),
            'species': s.get('species'),
            'dataset': s.get('dataset'),
            'source':  meta_source,
            'is_synthetic_negative': s.get('is_synthetic_negative', False),
            'pdb_id': s.get('pdb_id'),
        }
        metadata.append(meta_dict)

        chain   = (s.get('chain') or 'unknown').lower().strip()
        species = (s.get('species') or 'unknown').lower().strip()
        dataset = (s.get('dataset') or 'unknown').lower().strip()

        _chain_ids.append(_chain_vocab.get(chain, 3))
        _paired_ids.append(1 if s.get('paired', False) else 0)
        _species_ids.append(_species_vocab.get(species, 7))
        _dataset_ids.append(_dataset_vocab.get(dataset, 7))

    metadata_indices = {
        'chain_idx':   torch.tensor(_chain_ids,   dtype=torch.long),
        'paired_idx':  torch.tensor(_paired_ids,  dtype=torch.long),
        'species_idx': torch.tensor(_species_ids, dtype=torch.long),
        'dataset_idx': torch.tensor(_dataset_ids, dtype=torch.long),
    }
        
    synthetic_negative_mask = torch.tensor(
                    [
                        s.get("is_synthetic_negative", False)
                        for s in batch
                    ],
                    dtype=torch.bool,
                )

    return {
        # sequence
        "seq_tokens": seq_tokens,
        "seq_mask": seq_mask,
        "cdr_mask": cdr_mask,
        "cdr3_mask": cdr3_mask,
        "cdr12_mask": cdr12_mask,
        "contact_mask": contact_mask,
        "chain_position_mask": chain_position_mask,
        "cdr3_lengths":  cdr3_lengths if cdr_mask is not None else None,
        
        "seq_tokens_orig": seq_tokens_orig,
        "mlm_labels": mlm_labels,

        # structure
        "coords": coords,
        "coord_mask": coord_mask,
        "native_coords": coords,

        # antigen
        "antigen_coords": antigen_coords,
        "antigen_coord_mask": antigen_mask,
        "antigen_seq_tokens": antigen_seq_tokens,
        
        "synthetic_negative_mask": synthetic_negative_mask,
        
        # interface
        "interface_labels": interface_labels,
        "interface_mask": interface_mask,

        # task mask
        "task_mask": task_mask,
        "sample_task_mask": sample_task_mask,

        # metadata
        "metadata": metadata,
        "metadata_indices": metadata_indices,
        "source": batch[0].get("source", "unknown") if batch else "unknown",
        "pdb_ids": [s.get("pdb_id") or None for s in batch],
        "targets": [
                    s.get("target", None)
                    for s in batch
                ],
    }
