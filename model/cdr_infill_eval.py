"""
model/cdr_infill_eval.py
========================
CDR-infill evaluation matching protocols.

  1. AAR scored only on generated positions (h3_mask & gen_mask & valid_seq)
  2. Generation is truly AR: non-masked positions revealed from start,
     masked positions filled left-to-right in a single pass
  3. seq_tokens_orig always carries clean ground truth, never the growing gen
  4. contact_mask bug fixed: distances computed on masked atoms only,
     mapped back to residues via per-residue min
  5. Per-CDR detection uses stored cdr_type ids when available, falls back
     to transition-based detection with explicit chain separation
  6. Antigen shuffle adds sequence_change_rate metric
  7. AR NLL / perplexity added for masked positions
Usage:
    from model.cdr_infill_eval import cdr_infill_evaluation, print_cdr_infill_report
    results = cdr_infill_evaluation(model, rabd_loader, device)
    print_cdr_infill_report(results)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Optional, Tuple


MASK_IDX = 4
PAD_IDX  = 0
CDR_NAMES = ["H1", "H2", "H3", "L1", "L2", "L3"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _h3_heavy_mask(cdr3_mask, chain_pm, L):
    """(B, L) bool - CDR-H3 positions on the heavy chain."""
    if cdr3_mask is None or chain_pm is None:
        return None
    heavy = (chain_pm[:, :L] == 0)
    return cdr3_mask[:, :L].bool() & heavy


def _build_gen_mask(batch, mask_which: str, L: int) -> Optional[torch.Tensor]:
    """
    Return (B, L) bool tensor of positions to generate.

    mask_which:
        "cdr3_heavy"  AGForce: H3 only
        "all_cdrs"    AbX    : all 6 CDRs
        "full"        SCAT   : everything except START token
    """
    seq_mask  = batch["seq_mask"]
    cdr_mask  = batch.get("cdr_mask")
    cdr3_mask = batch.get("cdr3_mask")
    chain_pm  = batch.get("chain_position_mask")

    if mask_which == "cdr3_heavy":
        m = _h3_heavy_mask(cdr3_mask, chain_pm, L)
        return m

    if mask_which == "all_cdrs":
        if cdr_mask is None:
            return None
        return cdr_mask[:, :L].bool()

    # "full" - everything valid except position 0 (START)
    m = seq_mask[:, :L].clone()
    m[:, 0] = False
    return m


def _shuffle_antigen(batch, device):
    ag = batch.get("antigen_coords")
    if ag is None:
        return None

    B = ag.shape[0]

    if B < 2:
        return None

    perm = torch.roll(
        torch.arange(B, device=device),
        shifts=max(1, B // 2)
    )

    b2 = dict(batch)

    b2["antigen_coords"] = ag[perm]

    if batch.get("antigen_coord_mask") is not None:
        b2["antigen_coord_mask"] = batch["antigen_coord_mask"][perm]

    if batch.get("antigen_seq_tokens") is not None:
        b2["antigen_seq_tokens"] = batch["antigen_seq_tokens"][perm]

    return b2

@torch.no_grad()
def _ar_generate(model_raw, batch, gen_mask: torch.Tensor) -> torch.Tensor:
    """
    Greedy autoregressive generation.

    Non-masked positions are revealed from the start (framework / other CDRs).
    Masked positions are filled left-to-right in a single sequential pass.

    Fix 3: seq_tokens_orig always carries the CLEAN ground truth, so any
    internal conditioning that reads orig (geometry encoder chemistry path)
    sees real framework residues, not MASK tokens.  Only seq_tokens (the
    decoder input) carries the growing generation.
    """
    seq_orig = batch["seq_tokens_orig"]      # clean ground truth - never mutated
    seq_mask = batch["seq_mask"]
    B, L     = seq_orig.shape

    # Decoder input: start with revealed (non-masked) positions
    gen = seq_orig.clone()
    gen[gen_mask] = MASK_IDX
    gen[~seq_mask] = PAD_IDX
    
    coords_in = batch.get("coords")
    coord_mask_in = batch.get("coord_mask")

    if batch.get("_hide_ab_structure", False):
        coords_in = None
        coord_mask_in = None

    for p in range(1, L):
        upd = gen_mask[:, p] & seq_mask[:, p]   # positions to fill at step p
        if not upd.any():
            continue

        out = model_raw(
            seq_tokens          = gen,           # carries current generation state
            seq_tokens_orig     = seq_orig,      # Fix 3: always clean ground truth
            seq_mask            = seq_mask,
            cdr_mask            = batch.get("cdr_mask"),
            cdr3_mask           = batch.get("cdr3_mask"),
            coords = coords_in,
            coord_mask = coord_mask_in,
            # coords              = batch.get("coords"),
            # coord_mask          = batch.get("coord_mask"),
            antigen_coords      = batch.get("antigen_coords"),
            antigen_coord_mask  = batch.get("antigen_coord_mask"),
            antigen_seq_tokens  = batch.get("antigen_seq_tokens"),
            chain_mask          = batch.get("chain_position_mask"),
            cdr3_lengths        = batch.get("cdr3_lengths"),
            metadata=batch.get("metadata"),
            metadata_indices=batch.get("metadata_indices"),
            task_mask           = {"AR": True, "MLM": False,
                                   "GEO": False, "IF": False},
        )

        lg = out.get("logits_ar")              # (B, L-1, V)
        if lg is None or (p - 1) >= lg.shape[1]:
            continue

        gen[upd, p] = lg[:, p - 1, :].argmax(dim=-1)[upd]

    return gen

@torch.no_grad()
def _ar_nll_masked(model_raw, batch, gen_mask: torch.Tensor,
                   batch_shuf=None) -> dict:
    """
    Teacher-forced NLL on the positions that would be generated.
    Returns dict with keys: nll, ppl, nll_shuf, ppl_shuf, nll_gap.
    """
    seq_orig = batch["seq_tokens_orig"]
    seq_mask = batch["seq_mask"]
    B, L     = seq_orig.shape

    masked_in = seq_orig.clone()
    masked_in[gen_mask] = MASK_IDX

    def _fwd(b):
        coords_in = b.get("coords")
        coord_mask_in = b.get("coord_mask")

        if b.get("_hide_ab_structure", False):
            coords_in = None
            coord_mask_in = None
            
        out = model_raw(
            seq_tokens         = masked_in,
            seq_tokens_orig    = seq_orig,     # Fix 3: always clean
            seq_mask           = seq_mask,
            cdr_mask           = b.get("cdr_mask"),
            cdr3_mask          = b.get("cdr3_mask"),
            coords             = coords_in,
            coord_mask         = coord_mask_in,
            antigen_coords     = b.get("antigen_coords"),
            antigen_coord_mask = b.get("antigen_coord_mask"),
            antigen_seq_tokens = b.get("antigen_seq_tokens"),
            chain_mask         = b.get("chain_position_mask"),
            cdr3_lengths       = b.get("cdr3_lengths"),
            metadata           = b.get("metadata"),
            metadata_indices   = b.get("metadata_indices"),
            task_mask          = {"AR": True, "MLM": False,
                                  "GEO": False, "IF": False},
        )
        return out.get("logits_ar")

    logits = _fwd(batch)
    if logits is None:
        return {}

    # Align masks
    targets  = seq_orig[:, 1:].contiguous()          # (B, L-1)
    vm       = gen_mask[:, 1:].contiguous()[:, :targets.shape[1]]
    sm       = seq_mask[:, 1:].contiguous()[:, :targets.shape[1]]
    valid    = vm & sm & (targets != PAD_IDX)

    if not valid.any():
        return {}

    logits = logits[:, :targets.shape[1]]
    nll_tok = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
        ignore_index=PAD_IDX,
    ).reshape(targets.shape)

    nll  = nll_tok[valid].mean().item()
    ppl  = math.exp(min(nll, 20))
    out  = {"nll": nll, "ppl": ppl}

    if batch_shuf is not None:
        logits_s = _fwd(batch_shuf)
        if logits_s is not None:
            logits_s = logits_s[:, :targets.shape[1]]
            nll_s = F.cross_entropy(
                logits_s.reshape(-1, logits_s.shape[-1]),
                targets.reshape(-1),
                reduction="none",
                ignore_index=PAD_IDX,
            ).reshape(targets.shape)[valid].mean().item()
            out["nll_shuf"]  = nll_s
            out["ppl_shuf"]  = math.exp(min(nll_s, 20))
            out["nll_gap"]   = nll_s - nll      # positive = real antigen helps
    return out

def _contact_mask_residue(batch, L: int, cutoff: float = 8.0
                          ) -> Optional[torch.Tensor]:
    """
    (B, L) bool - antibody residues within `cutoff` Å of any antigen atom.

    Fix 5: distances computed only on valid (non-padded) atoms;
           result mapped back to residue level via per-residue minimum.
    Uses native antibody coords so contacts reflect experimental interface.
    """
    ab_coords = batch.get("coords")           # (B, L, N, 3)
    ag_coords = batch.get("antigen_coords")   # (B, M, N, 3)
    ab_mask_t = batch.get("coord_mask")       # (B, L, N) bool
    ag_mask_t = batch.get("antigen_coord_mask")

    if ab_coords is None or ag_coords is None:
        return None

    B    = ab_coords.shape[0]
    N_ab = ab_coords.shape[2]
    N_ag = ag_coords.shape[2]
    contact = torch.zeros(B, L, dtype=torch.bool, device=ab_coords.device)

    for b in range(B):
        ab_c = ab_coords[b, :L].float()         # (L, N, 3)
        ag_c = ag_coords[b].float()             # (M, N, 3)

        # Build per-residue validity for antibody
        if ab_mask_t is not None:
            ab_res_valid = ab_mask_t[b, :L].any(dim=-1)   # (L,) - residue has ≥1 atom
        else:
            ab_res_valid = torch.ones(L, dtype=torch.bool,
                                      device=ab_coords.device)

        # Flatten antigen atoms, apply mask
        ag_flat = ag_c.reshape(-1, 3)           # (M*N, 3)
        if ag_mask_t is not None:
            ag_am   = ag_mask_t[b].reshape(-1).bool()
            ag_flat = ag_flat[ag_am]            # (valid_ag_atoms, 3)
        if ag_flat.shape[0] == 0:
            continue

        # Per-residue minimum distance to any antigen atom
        # Process in chunks to stay within GPU memory
        min_dist = torch.full((L,), float("inf"), device=ab_coords.device)

        for l_start in range(0, L, 64):
            l_end  = min(l_start + 64, L)
            ab_seg = ab_c[l_start:l_end]        # (chunk, N, 3)

            # Apply per-atom mask
            if ab_mask_t is not None:
                ab_am_seg = ab_mask_t[b, l_start:l_end].bool()  # (chunk, N)
            else:
                ab_am_seg = torch.ones(ab_seg.shape[:2], dtype=torch.bool,
                                       device=ab_coords.device)

            ab_atoms = ab_seg[ab_am_seg]                  # (n_valid_atoms, 3)
            ab_res_idx = torch.arange(l_end - l_start, device=ab_coords.device)\
                            .unsqueeze(-1).expand_as(ab_am_seg)[ab_am_seg]

            if ab_atoms.shape[0] == 0:
                continue

            # (n_ab_atoms, n_ag_atoms)
            d = torch.cdist(ab_atoms.unsqueeze(0),
                            ag_flat.unsqueeze(0)).squeeze(0)   # (n_ab, n_ag)
            per_atom_min = d.min(dim=1).values                 # (n_ab,)

            # Scatter to residues: take minimum over atoms per residue
            seg_min = torch.full((l_end - l_start,), float("inf"),
                                 device=ab_coords.device)
            seg_min.scatter_reduce_(
                0, ab_res_idx, per_atom_min,
                reduce="amin", include_self=True
            )
            min_dist[l_start:l_end] = seg_min

        contact[b] = (min_dist < cutoff) & ab_res_valid

    return contact

def _per_cdr_masks(batch, L: int) -> Dict[str, torch.Tensor]:
    """
    Return {cdr_name: (B, L) bool} for H1-H3, L1-L3.

    Priority:
      1. batch["cdr_region_ids"] (B, L) int  codes: H1=0..L3=5  (future)
      2. transition-based detection on cdr_mask separated by chain
    """
    device   = batch["seq_tokens"].device
    B        = batch["seq_tokens"].shape[0]
    cdr_mask = batch.get("cdr_mask")
    chain_pm = batch.get("chain_position_mask")

    result = {n: torch.zeros(B, L, dtype=torch.bool, device=device)
              for n in CDR_NAMES}

    # Option 1: explicit region ids 
    cdr_ids = batch.get("cdr_region_ids")   # (B, L) int
    if cdr_ids is not None:
        code_map = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}
        for name, code in code_map.items():
            result[name] = (cdr_ids[:, :L] == code)
        return result

    # Option 2: transition-based, chain-separated
    if cdr_mask is None or chain_pm is None:
        return result

    cm   = cdr_mask[:, :L].bool()
    heavy = (chain_pm[:, :L] == 0)
    light = ~heavy

    for chain_flag, chain_mask_t, prefix, n_blocks in [
        (True,  heavy, "H", 3),
        (False, light, "L", 3),
    ]:
        cm_chain = cm & chain_mask_t           # CDR positions on this chain

        # Pad to detect transitions at both ends
        padded = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            cm_chain,
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ], dim=1)                              # (B, L+2)
        diff   = padded[:, 1:].long() - padded[:, :-1].long()   # (B, L+1)

        for b_idx in range(B):
            starts = (diff[b_idx] == 1 ).nonzero(as_tuple=False).flatten().tolist()
            ends   = (diff[b_idx] == -1).nonzero(as_tuple=False).flatten().tolist()
            n = min(len(starts), len(ends), n_blocks)
            for blk in range(n):
                name = f"{prefix}{blk + 1}"
                s, e = starts[blk], ends[blk]
                if e > s:
                    result[name][b_idx, s:e] = True

    return result


@torch.no_grad()
def cdr_infill_evaluation(
    model,
    loader,
    device,
    max_batches:   int   = 60,
    contact_cutoff: float = 8.0,
) -> dict:
    """
    Three evaluation modes in one pass:

    Mode          gen_mask              Compare to
    ----------    --------------------  --------------------------
    full          entire sequence       SCAT existing eval
    abx           all 6 CDRs            AbX Table 2
    agforce       CDR-H3 only           AGForce Table 1

    For each mode reports:
      aar_h3, aar_h3_shuf, gap_specific   (Fix 1: scored on generated positions)
      caar_h3                             (Fix 4+5: contact AAR)
      aar_all_cdr
      sequence_change_rate               (Fix 6: antigen blindness)
      nll, ppl, nll_shuf, ppl_shuf, nll_gap  (Fix 7: perplexity)
      per-CDR AAR: H1..L3               (Fix 6: clean block detection)
    """
    model_raw = _unwrap(model)
    model_raw.eval()

    MODES = ["full", "abx", "agforce"]

    # per-mode accumulators
    acc = {
        m: {
            "h3_c": 0, "h3_t": 0,
            "h3s_c": 0, "h3s_t": 0,        # shuffled
            "cdr_c": 0, "cdr_t": 0,
            "caar_c": 0, "caar_t": 0,
            "seq_change_sum": 0.0, "seq_change_n": 0,
            "nll_sum": 0.0, "nll_n": 0,
            "nll_shuf_sum": 0.0,
            **{f"cdr_{n}_c": 0 for n in CDR_NAMES},
            **{f"cdr_{n}_t": 0 for n in CDR_NAMES},
        }
        for m in MODES
    }

    n_batches = 0

    for raw_batch in loader:
        if raw_batch is None:
            continue
        if n_batches >= max_batches:
            break
        n_batches += 1

        batch = _to_device(raw_batch, device)

        # Guard required fields
        if batch.get("seq_tokens_orig") is None:
            batch["seq_tokens_orig"] = batch["seq_tokens"].clone()
        seq_orig = batch["seq_tokens_orig"]
        seq_mask = batch["seq_mask"]
        B, L     = seq_orig.shape

        # Pre-compute shared masks
        h3_mask    = _h3_heavy_mask(
            batch.get("cdr3_mask"), batch.get("chain_position_mask"), L)
        contact    = _contact_mask_residue(batch, L, contact_cutoff)
        cdr_blocks = _per_cdr_masks(batch, L)
        batch_shuf = _shuffle_antigen(batch, device)

        valid_base = seq_mask & (seq_orig != PAD_IDX) & (seq_orig != MASK_IDX)

        for mode in MODES:
            gen_mask = _build_gen_mask(batch, {
                "full":    "full",
                "abx":     "all_cdrs",
                "agforce": "cdr3_heavy",
            }[mode], L)

            if gen_mask is None:
                continue

            valid = valid_base & gen_mask

            # generate: real antigen
            gen      = _ar_generate(model_raw, batch, gen_mask)

            # generate: shuffled antigen
            gen_shuf = (
                        _ar_generate(model_raw, batch_shuf, gen_mask)
                        if batch_shuf is not None
                        else None
                    )

            a = acc[mode]

            # H3 AAR  (Fix 1)
            if h3_mask is not None:
                h3_valid = h3_mask & valid
                if h3_valid.any():
                    a["h3_c"] += int(((gen == seq_orig) & h3_valid).sum())
                    a["h3_t"] += int(h3_valid.sum())
                    a["h3s_c"] += int(((gen_shuf == seq_orig) & h3_valid).sum())
                    a["h3s_t"] += int(h3_valid.sum())

            # All-CDR AAR  (Fix 1)
            cdr_mask_t = batch.get("cdr_mask")
            if cdr_mask_t is not None:
                cdr_valid = cdr_mask_t[:, :L].bool() & valid
                if cdr_valid.any():
                    a["cdr_c"] += int(((gen == seq_orig) & cdr_valid).sum())
                    a["cdr_t"] += int(cdr_valid.sum())

            # CAAR  (Fix 4+5)
            if contact is not None and h3_mask is not None:
                caar_valid = h3_mask & contact & valid
                if caar_valid.any():
                    a["caar_c"] += int(((gen == seq_orig) & caar_valid).sum())
                    a["caar_t"] += int(caar_valid.sum())

            # Sequence change rate  (Fix 6)
            if h3_mask is not None and gen_shuf is not None:
                h3v = h3_mask & gen_mask & seq_mask
                if h3v.any():
                    changed = (gen != gen_shuf).float()
                    a["seq_change_sum"] += changed[h3v].mean().item()
                    a["seq_change_n"]   += 1

            # Per-CDR AAR  (Fix 6)
            for cdr_name, blk_mask in cdr_blocks.items():
                blk_valid = blk_mask & valid
                if blk_valid.any():
                    a[f"cdr_{cdr_name}_c"] += int(((gen == seq_orig) & blk_valid).sum())
                    a[f"cdr_{cdr_name}_t"] += int(blk_valid.sum())

            # NLL / perplexity  (Fix 7)
            nll_metrics = _ar_nll_masked(model_raw, batch, gen_mask, batch_shuf)
            if "nll" in nll_metrics:
                a["nll_sum"] += nll_metrics["nll"]
                a["nll_n"]   += 1
            if "nll_shuf" in nll_metrics:
                a["nll_shuf_sum"] += nll_metrics["nll_shuf"]

    # aggregate 
    def _r(c, t):
        return c / max(t, 1)

    def _ppl(s, n):
        return math.exp(min(s / max(n, 1), 20))

    results = {}
    for mode in MODES:
        a  = acc[mode]
        px = mode

        results[f"{px}_aar_h3"]       = _r(a["h3_c"],   a["h3_t"])
        results[f"{px}_aar_h3_shuf"]  = _r(a["h3s_c"],  a["h3s_t"])
        results[f"{px}_gap_specific"] = results[f"{px}_aar_h3"] - results[f"{px}_aar_h3_shuf"]
        results[f"{px}_aar_all_cdr"]  = _r(a["cdr_c"],  a["cdr_t"])
        results[f"{px}_caar_h3"]      = _r(a["caar_c"], a["caar_t"])
        results[f"{px}_n_h3_pos"]     = a["h3_t"]
        results[f"{px}_n_contact"]    = a["caar_t"]

        if a["seq_change_n"] > 0:
            results[f"{px}_seq_change_rate"] = (
                a["seq_change_sum"] / a["seq_change_n"])

        if a["nll_n"] > 0:
            nll_avg = a["nll_sum"] / a["nll_n"]
            results[f"{px}_masked_nll"]      = nll_avg
            results[f"{px}_masked_ppl"]      = math.exp(min(nll_avg, 20))
            if a["nll_shuf_sum"] > 0:
                ns = a["nll_shuf_sum"] / a["nll_n"]
                results[f"{px}_masked_nll_shuf"] = ns
                results[f"{px}_masked_ppl_shuf"] = math.exp(min(ns, 20))
                results[f"{px}_masked_nll_gap"]  = ns - nll_avg

        for cdr_name in CDR_NAMES:
            c = a[f"cdr_{cdr_name}_c"]
            t = a[f"cdr_{cdr_name}_t"]
            if t > 0:
                results[f"{px}_aar_{cdr_name}"] = _r(c, t)

    results["n_batches_evaluated"] = n_batches
    return results

# Pretty printer
def print_cdr_infill_report(results: dict):
    W = 80

    def fmt(x, pct=False):
        if x != x:          # nan
            return "   n/a "
        if pct:
            return f"{x*100:6.2f}%"
        return f"{x:.4f}"

    print("\n" + "=" * W)
    print("CDR INFILL EVALUATION - PROTOCOL COMPARISON")
    print(f"  (batches evaluated: {results.get('n_batches_evaluated', '?')})")
    print("=" * W)

    hdr = (f"{'Mode':<24}  {'AAR-H3':>8}  {'AAR-shuf':>9}  "
           f"{'gap':>7}  {'CAAR':>7}  {'CDR-all':>8}  "
           f"{'Δseq':>6}  {'PPL':>7}")
    print(hdr)
    print("-" * W)

    mode_labels = {
        "full":    "SCAT full-Fv (ours)",
        "abx":     "AbX-style (all CDRs)",
        "agforce": "AGForce-style (H3 only)",
    }

    for mode, label in mode_labels.items():
        px = mode
        row = (
            f"{label:<24}  "
            f"{fmt(results.get(f'{px}_aar_h3',     float('nan'))):>8}  "
            f"{fmt(results.get(f'{px}_aar_h3_shuf',float('nan'))):>9}  "
            f"{fmt(results.get(f'{px}_gap_specific',float('nan'))):>7}  "
            f"{fmt(results.get(f'{px}_caar_h3',    float('nan'))):>7}  "
            f"{fmt(results.get(f'{px}_aar_all_cdr',float('nan'))):>8}  "
            f"{fmt(results.get(f'{px}_seq_change_rate', float('nan'))):>6}  "
            f"{fmt(results.get(f'{px}_masked_ppl',        float('nan'))):>7}"
        )
        print(row)

    # NLL gap table
    print()
    print(f"{'Mode':<24}  {'NLL-real':>9}  {'NLL-shuf':>9}  {'NLL-gap':>8}  {'PPL-real':>9}")
    print("-" * W)
    for mode, label in mode_labels.items():
        px = mode
        row = (
            f"{label:<24}  "
            f"{fmt(results.get(f'{px}_masked_nll',      float('nan'))):>9}  "
            f"{fmt(results.get(f'{px}_masked_nll_shuf', float('nan'))):>9}  "
            f"{fmt(results.get(f'{px}_masked_nll_gap',  float('nan'))):>8}  "
            f"{fmt(results.get(f'{px}_masked_ppl',      float('nan'))):>9}"
        )
        print(row)

    # Per-CDR AAR Table 2
    print()
    print("-" * W)
    print("PER-CDR AAR  (AbX / AGForce Table 2 style)")
    hdr2 = (f"{'CDR':<8}  {'SCAT-full':>10}  "
            f"{'AbX-style':>10}  {'AGForce':>10}")
    print(hdr2)
    print("-" * W)
    for cdr in CDR_NAMES:
        fv  = results.get(f"full_aar_{cdr}",    float("nan"))
        av  = results.get(f"abx_aar_{cdr}",     float("nan"))
        gv  = results.get(f"agforce_aar_{cdr}", float("nan"))
        print(f"{cdr:<8}  {fmt(fv, pct=True):>10}  "
              f"{fmt(av, pct=True):>10}  {fmt(gv, pct=True):>10}")

    print("=" * W + "\n")