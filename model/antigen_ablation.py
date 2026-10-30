# model/antigen_ablation.py (old but we are using in training for [Antigen Recover and RAbD at epoch end from run_train]
"""
Antigen Ablation Controls for SCAT.

Three conditions run on the SAbDab validation set each val_check_interval:

  full        - normal forward pass (baseline)
  no_ag       - antigen_coords = None, antigen_seq_tokens = None
  shuffled_ag - antigen_coords cyclically shifted within the batch
  random_ag   - antigen_coords replaced with Gaussian noise (same shape)

Metrics reported for each condition:
  ar_ppl      - AR perplexity on full sequence (nats)
  cdr3_ppl    - AR perplexity on CDR3 positions only (nats)

Gap metrics (vs full baseline):
  ppl_gap_no_ag, ppl_gap_shuffled_ag, ppl_gap_random_ag
  cdr3_ppl_gap_no_ag, cdr3_ppl_gap_shuffled_ag, cdr3_ppl_gap_random_ag

Interpretation guide (after Phase-3 training):
  ppl_gap_no_ag       > 0.05   -> antigen presence is used
  ppl_gap_shuffled_ag > 0.02   -> antigen identity is used (specificity)
  cdr3_ppl_gap_no_ag  > 0.10   -> CDR3 is the most antigen-sensitive region

  All gaps ~= 0 -> antigen conditioning inactive; check antigen_xattn_gate value.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# Maximum number of SAbDab validation batches to run ablation on.
# Ablation does 4× forward passes, so kept this low to avoid dominating val time.
_MAX_ABLATION_BATCHES = 50
_IGNORE_IDX           = -100
_PAD_IDX              = 0
_MASK_IDX             = 4


class AntigenAblation:
    """
    Runs four forward passes per batch (full / no_ag / shuffled_ag / random_ag)
    and accumulates AR perplexity on all positions and CDR3 positions.

    Designed to run inside trainer.validate() on the SAbDab val loader only.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 max_batches: int = _MAX_ABLATION_BATCHES):
        self.model       = model
        self.device      = device
        self.max_batches = max_batches

    # Public API
    @torch.no_grad()
    def cdr3_recovery(self, sabdab_val_loader, max_batches: int = 4) -> Dict[str, float]:
        """
        Lean Mod-1 scorecard: greedy H3 infill with antibody backbone held fixed,
        under full vs shuffled antigen. Reports AbX-comparable Loop AAR (aar_h3_full)
        and the gap that scores antigen-identity use (aar_h3_gap_specific).

        Only heavy CDR3, only two conditions - keep max_batches small (≤4) and run
        every Nth validation, not every one.
        """
        from collections import defaultdict
        _was_training = self.model.training
        self.model.eval()
        acc = defaultdict(lambda: {"correct": 0, "total": 0})

        for bi, batch in enumerate(sabdab_val_loader):
            if bi >= max_batches: break
            if batch is None or batch.get("antigen_coords") is None: continue
            
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(self.device)

            batch = self._filter_real_pairs(batch)
            if batch is None:
                continue
            if batch.get("antigen_coords") is None or batch["antigen_coords"].shape[0] == 0:
                continue

            true     = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
            cdr3_b   = batch.get("cdr3_mask")
            cdr_b    = batch.get("cdr_mask")
            seq_b    = batch.get("seq_mask")
            chain    = batch.get("chain_position_mask")
            if true is None or cdr3_b is None or seq_b is None or chain is None:
                continue
            cdr3_b, seq_b = cdr3_b.bool(), seq_b.bool()
            h3_b = cdr3_b & (chain == 0) # heavy CDR3 only
            if not h3_b.any(): continue

            ag, agm, ags = batch["antigen_coords"], batch.get("antigen_coord_mask"), batch.get("antigen_seq_tokens")
            targets = batch.get("targets")
            perm = self._cross_class_perm(
                ag.shape[0],
                targets,
            )

            conditions = {
                "full": (ag, agm, ags),
                "shuffled_ag": (
                    self._apply_perm(ag, perm),
                    self._apply_perm(agm, perm),
                    self._apply_perm(ags, perm),
                ),
            }

            for cname, (ac, acm, asts) in conditions.items():
                gen = true.clone()              # start from full ground truth
                gen[h3_b] = _MASK_IDX           # mask only H3
                gen[~seq_b] = 0

                geom_gen = true.clone()
                geom_gen[cdr_b.bool() if cdr_b is not None else cdr3_b] = 4
                
                # Generate the entire antibody autoregressively
                max_len = min(
                    int(seq_b.sum(dim=1).max().item()),
                    true.shape[1]
                )
                cols = range(1, max_len)

                for p in cols:
                    out = self.model(
                        seq_tokens=gen,
                        seq_tokens_orig=gen,
                        # seq_tokens_orig=true,
                        geom_seq_tokens=geom_gen,
                        seq_mask=batch.get("seq_mask"),
                        cdr_mask=batch.get("cdr_mask"),
                        coords=batch.get("coords"),
                        coord_mask=batch.get("coord_mask"),
                        antigen_coords=ac,
                        antigen_coord_mask=acm,
                        antigen_seq_tokens=asts,
                        task_mask={"AR": True, "MLM": False, "GEO": False, "IF": False},
                        metadata=batch.get("metadata"),
                        metadata_indices=batch.get("metadata_indices"),
                        chain_mask=chain,
                        cdr3_lengths=batch.get("cdr3_lengths"),
                        cdr3_mask=batch.get("cdr3_mask"),
                    )

                    lg = out.get("logits_ar")
                    if lg is None or (p - 1) >= lg.shape[1]:
                        continue

                    upd = h3_b[:, p]
                    if upd.any():
                        gen[upd, p] = lg[:, p - 1, :].argmax(dim=-1)[upd]
                
                valid = h3_b & seq_b & (true != 0)
                acc[cname]["correct"] += int(((gen == true) & valid).sum().item())
                acc[cname]["total"]   += int(valid.sum().item())

        def aar(c): return acc[c]["correct"] / acc[c]["total"] if acc[c]["total"] > 0 else 0.0
        m = {
            "aar_h3_full":         aar("full"),
            "aar_h3_shuffled_ag":  aar("shuffled_ag"),
        }
        m["aar_h3_gap_specific"] = m["aar_h3_full"] - m["aar_h3_shuffled_ag"]
        if _was_training:
            self.model.train()
        return m
    
    @torch.no_grad()
    def antigen_classification_probe(
        self, sabdab_val_loader, max_batches: int = 10
    ) -> dict:
        """
        InfoNCE-style antigen classification: can the predicted CDR3 probability
        distribution identify the correct antigen from the batch?

        Accuracy ≈ 1/N  →  antigen blind (random chance).
        Accuracy >> 1/N  →  antigen identity survives into predictions.

        Two probes run in parallel:
          (a) soft_seq vs antigen_pooled  - cross-space cosine similarity.
          (b) soft_seq vs soft_seq        - within-distribution InfoNCE
                                           (diagonal = real pair).
        """
        _was_training = self.model.training
        self.model.eval()

        all_soft_emb   = []   # mean over CDR3 positions of predicted probs [V]
        all_ag_pooled  = []   # antigen_pooled from model output [d_model]
        all_soft_shuf  = []   # same but under shuffled antigen

        for bi, batch in enumerate(sabdab_val_loader):
            if bi >= max_batches:
                break
            if batch is None or batch.get("antigen_coords") is None:
                continue

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(self.device)

            batch = self._filter_real_pairs(batch)
            if batch is None or batch.get("antigen_coords") is None:
                continue

            ag, agm, ags = (
                batch["antigen_coords"],
                batch.get("antigen_coord_mask"),
                batch.get("antigen_seq_tokens"),
            )
            perm = self._cross_class_perm(ag.shape[0], batch.get("targets"))

            def _fwd(ac, acm, asts):
                clean = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
                geom  = clean.clone()
                cdr_b = batch.get("cdr_mask")
                if cdr_b is not None:
                    geom[cdr_b.bool()] = 4
                return self.model(
                    seq_tokens=clean, seq_tokens_orig=clean,
                    geom_seq_tokens=geom,
                    seq_mask=batch.get("seq_mask"),
                    cdr_mask=batch.get("cdr_mask"),
                    coords=batch.get("coords"),
                    coord_mask=batch.get("coord_mask"),
                    antigen_coords=ac, antigen_coord_mask=acm,
                    antigen_seq_tokens=asts,
                    task_mask={"AR": True, "MLM": False, "GEO": False, "IF": False},
                    metadata=batch.get("metadata"),
                    metadata_indices=batch.get("metadata_indices"),
                    chain_mask=batch.get("chain_position_mask"),
                    cdr3_lengths=batch.get("cdr3_lengths"),
                    cdr3_mask=batch.get("cdr3_mask"),
                )

            out_full = _fwd(ag, agm, ags)
            out_shuf = _fwd(
                self._apply_perm(ag, perm),
                self._apply_perm(agm, perm),
                self._apply_perm(ags, perm),
            )

            logits_full = out_full.get("logits_ar")
            logits_shuf = out_shuf.get("logits_ar")
            ag_pooled   = out_full.get("antigen_pooled")
            cdr3_mask   = batch.get("cdr3_mask")

            if logits_full is None or cdr3_mask is None:
                continue

            probs_full = torch.softmax(logits_full.float(), dim=-1)   # (B, L-1, V)
            probs_shuf = (
                torch.softmax(logits_shuf.float(), dim=-1)
                if logits_shuf is not None else None
            )

            # shift cdr3_mask by 1 to align with AR logits
            cdr3_shifted = cdr3_mask[:, 1:logits_full.shape[1] + 1].bool()

            for b in range(probs_full.shape[0]):
                m = cdr3_shifted[b]
                if m.sum() < 3:
                    continue

                soft = probs_full[b][m].mean(dim=0).cpu()        # (V,)
                all_soft_emb.append(soft)

                if ag_pooled is not None:
                    all_ag_pooled.append(ag_pooled[b].cpu())

                if probs_shuf is not None:
                    soft_s = probs_shuf[b][m].mean(dim=0).cpu()
                    all_soft_shuf.append(soft_s)

        if len(all_soft_emb) < 4:
            if _was_training:
                self.model.train()
            return {"antigen_cls_n_samples": len(all_soft_emb)}

        N = len(all_soft_emb)
        soft = torch.stack(all_soft_emb)          # (N, V)
        soft = soft / (soft.norm(dim=1, keepdim=True) + 1e-8)

        results = {"antigen_cls_n_samples": N,
                   "antigen_cls_random_baseline": 1.0 / N}

        # ── probe (b): within soft-seq InfoNCE (self-similarity) ─────────────
        # Diagonal = same sample under real antigen (positive pair).
        # Off-diagonal = different samples (implicit negatives).
        sim_self = soft @ soft.T     # (N, N)
        # For each row, how many off-diagonal entries exceed the diagonal?
        ranks_self = (sim_self > sim_self.diag().unsqueeze(1)).sum(dim=1).float()
        results["antigen_cls_self_top1"] = float((ranks_self == 0).float().mean())
        results["antigen_cls_self_top5"] = float((ranks_self < 5).float().mean())

        # ── probe against shuffled antigen ────────────────────────────────────
        if len(all_soft_shuf) == N:
            soft_s = torch.stack(all_soft_shuf)
            soft_s = soft_s / (soft_s.norm(dim=1, keepdim=True) + 1e-8)
            # cosine sim between real-antigen soft-seq and shuffled-antigen soft-seq (diagonal)
            diag_cos = (soft * soft_s).sum(dim=1)   # (N,)
            results["antigen_cls_real_vs_shuf_cos_mean"] = float(diag_cos.mean())
            results["antigen_cls_real_vs_shuf_cos_std"]  = float(diag_cos.std())
            # discrimination: does real beat shuffled for the same sample?
            sim_real_to_all_shuf = soft @ soft_s.T   # (N, N) - row i = sim(real_i, shuf_j)
            real_vs_shuf_rank = (
                sim_real_to_all_shuf > sim_real_to_all_shuf.diag().unsqueeze(1)
            ).sum(dim=1).float()
            results["antigen_cls_real_beats_shuf_top1"] = float(
                (real_vs_shuf_rank == 0).float().mean()
            )

        # rank test using ag_pooled internal similarity instead:
        if len(all_ag_pooled) == N:
            ag_p = torch.stack(all_ag_pooled)          # (N, d_model)
            ag_p = ag_p / (ag_p.norm(dim=1, keepdim=True) + 1e-8)
            # Antigen self-similarity: how well does antigen_pooled cluster by identity?
            # # antigen_pooled comes from antigen_encoder — measures antigen representation diversity.
            sim_ag = ag_p @ ag_p.T                     # (N, N)
            off_diag = sim_ag[~torch.eye(N, dtype=torch.bool)].mean()
            results["antigen_pooled_mean_offdiag_cos"] = float(off_diag)


        if _was_training:
            self.model.train()
        return results

    @torch.no_grad()
    def path_isolation(
        self, sabdab_val_loader, max_batches: int = 10
    ) -> dict:
        """
        Isolates contribution of antigen cross-attention (Path B).

        v2 has no Path A gate (cond_proj has no gate).
        Two conditions:
          full          — normal forward (antigen xattn active)
          xattn_disabled — antigen_xattn_gate forced to -20 (sigmoid≈0, xattn off)

        Reports CDR3 PPL and H3 AAR for each condition.
        """
        _was_training = self.model.training
        self.model.eval()

        _m = self.model.module if hasattr(self.model, 'module') else self.model

        orig_ag = _m.antigen_xattn_gate.item()
        DISABLED = -20.0  # sigmoid(-20) ≈ 2e-9 ≈ 0

        gate_configs = {
            "full":           orig_ag,
            "xattn_disabled": DISABLED,
        }

        results = {}

        for cond_name, ag_val in gate_configs.items():
            with torch.no_grad():
                _m.antigen_xattn_gate.fill_(ag_val)

            acc = {"cdr3_nll": 0.0, "cdr3_n": 0,
                   "h3_correct": 0,  "h3_total": 0}

            for bi, batch in enumerate(sabdab_val_loader):
                if bi >= max_batches:
                    break
                if batch is None or batch.get("antigen_coords") is None:
                    continue

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if torch.is_tensor(vv):
                                v[kk] = vv.to(self.device)

                batch = self._filter_real_pairs(batch)
                if batch is None or batch.get("antigen_coords") is None:
                    continue

                nll, n, cdr3_nll, cdr3_n = self._batch_ar_nll(
                    batch,
                    batch["antigen_coords"],
                    batch.get("antigen_coord_mask"),
                    batch.get("antigen_seq_tokens"),
                )
                if nll is not None:
                    acc["cdr3_nll"] += cdr3_nll
                    acc["cdr3_n"]   += cdr3_n

                true   = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
                cdr3_b = batch.get("cdr3_mask")
                seq_b  = batch.get("seq_mask")
                chain  = batch.get("chain_position_mask")
                cdr_b  = batch.get("cdr_mask")
                if true is None or cdr3_b is None or seq_b is None or chain is None:
                    continue

                cdr3_b, seq_b = cdr3_b.bool(), seq_b.bool()
                h3_b = cdr3_b & (chain == 0)
                if not h3_b.any():
                    continue

                ag  = batch["antigen_coords"]
                agm = batch.get("antigen_coord_mask")
                ags = batch.get("antigen_seq_tokens")

                gen = true.clone()       # reveal framework + other CDRs (AGForce protocol)
                gen[h3_b] = _MASK_IDX   # mask only CDR-H3
                gen[~seq_b] = 0
                geom_gen = true.clone()
                if cdr_b is not None:
                    geom_gen[cdr_b.bool()] = 4

                max_len = min(int(seq_b.sum(dim=1).max().item()), true.shape[1])
                for p in range(1, max_len):
                    out = self.model(
                        seq_tokens=gen, seq_tokens_orig=gen,
                        geom_seq_tokens=geom_gen,
                        seq_mask=batch.get("seq_mask"),
                        cdr_mask=batch.get("cdr_mask"),
                        coords=batch.get("coords"),
                        coord_mask=batch.get("coord_mask"),
                        antigen_coords=ag, antigen_coord_mask=agm,
                        antigen_seq_tokens=ags,
                        task_mask={"AR": True, "MLM": False, "GEO": False, "IF": False},
                        metadata=batch.get("metadata"),
                        metadata_indices=batch.get("metadata_indices"),
                        chain_mask=chain,
                        cdr3_lengths=batch.get("cdr3_lengths"),
                        cdr3_mask=batch.get("cdr3_mask"),
                    )
                    lg = out.get("logits_ar")
                    if lg is None or (p - 1) >= lg.shape[1]:
                        continue
                    upd = seq_b[:, p]
                    gen[upd, p] = lg[:, p - 1, :].argmax(dim=-1)[upd]

                valid = h3_b & seq_b & (true != 0)
                acc["h3_correct"] += int(((gen == true) & valid).sum())
                acc["h3_total"]   += int(valid.sum())

            cdr3_ppl = math.exp(acc["cdr3_nll"] / acc["cdr3_n"]) \
                       if acc["cdr3_n"] > 0 else float('nan')
            h3_aar   = acc["h3_correct"] / acc["h3_total"] \
                       if acc["h3_total"] > 0 else float('nan')
            results[cond_name] = {"cdr3_ppl": cdr3_ppl, "h3_aar": h3_aar}

        # Restore original gate value
        with torch.no_grad():
            _m.antigen_xattn_gate.fill_(orig_ag)

        if _was_training:
            self.model.train()

        print("\n=== PATH ISOLATION ABLATION ===")
        print(f"  {'Condition':<20} {'CDR3-PPL':>10} {'H3-AAR':>10}")
        print(f"  {'-'*42}")
        for cname, m in results.items():
            print(f"  {cname:<20} {m['cdr3_ppl']:>10.4f} {m['h3_aar']:>10.4f}")
        full = results.get("full", {})
        for cname in ("xattn_disabled",):
            r = results.get(cname, {})
            ppl_d = r.get("cdr3_ppl", float('nan')) - full.get("cdr3_ppl", float('nan'))
            aar_d = r.get("h3_aar",   float('nan')) - full.get("h3_aar",   float('nan'))
            print(f"  {cname:<20} ΔPPL={ppl_d:+.4f}  ΔAAR={aar_d:+.4f}")

        return results
    
    @torch.no_grad()
    def length_stratified_aar(
        self, sabdab_val_loader, max_batches: int = 8
    ) -> dict:
        """
        AAR broken down by CDR-H3 length buckets.
        Reveals whether antigen conditioning helps more on longer loops
        where the germline prior is weaker.

        Buckets: short (≤10), medium (11-14), long (≥15)
        """
        _was_training = self.model.training
        self.model.eval()

        buckets = {
            "short":  {"range": (0,  10), "full": [0,0], "shuf": [0,0]},
            "medium": {"range": (11, 14), "full": [0,0], "shuf": [0,0]},
            "long":   {"range": (15, 99), "full": [0,0], "shuf": [0,0]},
        }

        for bi, batch in enumerate(sabdab_val_loader):
            if bi >= max_batches:
                break
            if batch is None or batch.get("antigen_coords") is None:
                continue

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(self.device)

            batch = self._filter_real_pairs(batch)
            if batch is None or batch.get("antigen_coords") is None:
                continue

            true   = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
            cdr3_b = batch.get("cdr3_mask")
            seq_b  = batch.get("seq_mask")
            chain  = batch.get("chain_position_mask")
            cdr_b  = batch.get("cdr_mask")
            if true is None or cdr3_b is None or seq_b is None or chain is None:
                continue

            cdr3_b, seq_b = cdr3_b.bool(), seq_b.bool()
            h3_mask = cdr3_b & (chain == 0)

            ag  = batch["antigen_coords"]
            agm = batch.get("antigen_coord_mask")
            ags = batch.get("antigen_seq_tokens")
            perm = self._cross_class_perm(ag.shape[0], batch.get("targets"))

            for cond_name, (ac, acm, ast) in [
                ("full", (ag, agm, ags)),
                ("shuf", (self._apply_perm(ag, perm),
                        self._apply_perm(agm, perm),
                        self._apply_perm(ags, perm))),
            ]:
                gen = true.clone()       # reveal framework + other CDRs (AGForce protocol)
                gen[h3_b] = _MASK_IDX   # mask only CDR-H3
                gen[~seq_b] = 0
                geom_gen = true.clone()
                if cdr_b is not None:
                    geom_gen[cdr_b.bool()] = 4

                max_len = min(int(seq_b.sum(dim=1).max().item()), true.shape[1])
                for p in range(1, max_len):
                    out = self.model(
                        seq_tokens=gen, seq_tokens_orig=gen,
                        geom_seq_tokens=geom_gen,
                        seq_mask=batch.get("seq_mask"),
                        cdr_mask=batch.get("cdr_mask"),
                        coords=batch.get("coords"),
                        coord_mask=batch.get("coord_mask"),
                        antigen_coords=ac, antigen_coord_mask=acm,
                        antigen_seq_tokens=ast,
                        task_mask={"AR": True, "MLM": False, "GEO": False, "IF": False},
                        metadata=batch.get("metadata"),
                        metadata_indices=batch.get("metadata_indices"),
                        chain_mask=chain,
                        cdr3_lengths=batch.get("cdr3_lengths"),
                        cdr3_mask=batch.get("cdr3_mask"),
                    )
                    lg = out.get("logits_ar")
                    if lg is None or (p - 1) >= lg.shape[1]:
                        continue
                    upd = seq_b[:, p]
                    gen[upd, p] = lg[:, p - 1, :].argmax(dim=-1)[upd]

                # Score per sample, bucketed by H3 length
                for b in range(true.shape[0]):
                    h3_len = int(h3_mask[b].sum().item())
                    valid  = h3_mask[b] & seq_b[b] & (true[b] != 0)
                    if not valid.any():
                        continue
                    correct = int(((gen[b] == true[b]) & valid).sum())
                    total   = int(valid.sum())
                    for bname, bdata in buckets.items():
                        lo, hi = bdata["range"]
                        if lo <= h3_len <= hi:
                            bdata[cond_name][0] += correct
                            bdata[cond_name][1] += total

        print("\n=== LENGTH-STRATIFIED H3 AAR ===")
        print(f"  {'Bucket':<10} {'Length':>8} {'AAR-full':>10} "
            f"{'AAR-shuf':>10} {'gap':>10} {'n_pos':>8}")
        print(f"  {'-'*56}")

        results = {}
        for bname, bdata in buckets.items():
            lo, hi    = bdata["range"]
            aar_full  = bdata["full"][0]  / bdata["full"][1]  if bdata["full"][1]  > 0 else float('nan')
            aar_shuf  = bdata["shuf"][0]  / bdata["shuf"][1]  if bdata["shuf"][1]  > 0 else float('nan')
            gap       = aar_full - aar_shuf if not (aar_full != aar_full) else float('nan')
            lrange    = f"{lo}-{hi}" if hi < 99 else f"≥{lo}"
            print(f"  {bname:<10} {lrange:>8} {aar_full:>10.4f} "
                f"{aar_shuf:>10.4f} {gap:>+10.4f} {bdata['full'][1]:>8}")
            results[bname] = {"aar_full": aar_full, "aar_shuf": aar_shuf,
                            "gap": gap, "n": bdata["full"][1]}

        if _was_training:
            self.model.train()
        return results
    
    @torch.no_grad()
    def run(
        self,
        sabdab_val_loader,
        global_step: int = 0,
    ) -> Dict[str, float]:
        """
        Iterate through up to max_batches of the SAbDab val loader.
        Returns a flat dict of scalar metrics.
        """
        self.model.eval()

        accum = {
            cond: {"nll_sum": 0.0, "n_tokens": 0,
                   "cdr3_nll_sum": 0.0, "cdr3_tokens": 0}
            for cond in ("full", "no_ag", "shuffled_ag", "random_ag")
        }

        for batch_idx, batch in enumerate(sabdab_val_loader):
            if batch_idx >= self.max_batches:
                break
            if batch is None:
                continue
            if batch.get("antigen_coords") is None:
                continue

            # Move to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(self.device)
            
            batch = self._filter_real_pairs(batch)
            if batch is None:
                continue
            if batch.get("antigen_coords") is None or batch["antigen_coords"].shape[0] == 0:
                continue

            ag_coords      = batch["antigen_coords"]
            ag_coord_mask  = batch.get("antigen_coord_mask")
            ag_seq_tokens  = batch.get("antigen_seq_tokens")
            targets = batch.get("targets")
            
            perm = self._cross_class_perm(
                    ag_coords.shape[0],
                    targets,
                )

            # four conditions 
            conditions = {
                "full":        (ag_coords, ag_coord_mask, ag_seq_tokens),
                "no_ag":       (None, None, None),
                "shuffled_ag": (
                        self._apply_perm(ag_coords, perm),
                        self._apply_perm(ag_coord_mask, perm),
                        self._apply_perm(ag_seq_tokens, perm),
                    ),
                "random_ag": (self._randomize(ag_coords), ag_coord_mask, None,),
            }

            for cond_name, (ac, acm, asts) in conditions.items():
                nll, n, cdr3_nll, cdr3_n = self._batch_ar_nll(
                    batch, ac, acm, asts
                )
                if nll is not None:
                    accum[cond_name]["nll_sum"]      += nll
                    accum[cond_name]["n_tokens"]     += n
                    accum[cond_name]["cdr3_nll_sum"] += cdr3_nll
                    accum[cond_name]["cdr3_tokens"]  += cdr3_n

        return self._build_metrics(accum)
    
    @torch.no_grad()
    def per_cdr_metrics(
        self,
        sabdab_val_loader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """
        Per-CDR amino-acid recovery (AAR) and Ca-RMSD for the full-antigen
        vs no-antigen conditions, replicating the structure of AbX Table 2.
 
        Regions reported:
        Key            Mask used                      Paper equivalent
        cdr3_heavy     cdr3_mask & chain==0           H3  (exact match)
        cdr3_light     cdr3_mask & chain==1           L3  (exact match)
        all_cdr        cdr_mask  (both chains)        H1+H2+H3+L1+L2+L3
        all_cdr_h      cdr_mask  & chain==0           H1+H2+H3 pooled
        all_cdr_l      cdr_mask  & chain==1           L1+L2+L3 pooled
 
        To unlock individual H1/H2/L1/L2 rows (full Table 2), add a
        "cdr_region_ids" tensor (B, L, int, codes 0-5) in your collate_fn
        and uncomment the block inside _per_cdr_regions().
 
        Metrics returned (flat dict of floats):
            aar_<region>_full      / aar_<region>_no_ag
            rmsd_<region>_full     / rmsd_<region>_no_ag   (needs native_coords)
            aar_<region>_ag_gap    =  full - no_ag
 
        Ca-RMSD is Kabsch-aligned per sample over the CDR region, matching
        the AbX/DiffAb evaluation protocol.  RMSD keys are only emitted when
        the batch contains "native_coords" (B, L, 4, 3) - the ground-truth
        backbone atom coordinates.  If your dataset doesn't supply it yet,
        AAR metrics are still fully reported.
        """
        _was_training = self.model.training
        self.model.eval()
 
        from collections import defaultdict
        acc: Dict[str, Dict] = {
            cond: defaultdict(lambda: {"correct": 0, "total": 0,
                                       "sq_err": 0.0,  "n_res":  0})
            for cond in ("full", "no_ag")
        }
 
        for bi, batch in enumerate(sabdab_val_loader):
            if bi >= max_batches:
                break
            if batch is None:
                continue
 
            # ── device transfer ──────────────────────────────────────────
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(self.device)
 
            batch = self._filter_real_pairs(batch)
            if batch is None:
                continue
 
            # ── region masks for this batch ──────────────────────────────
            regions = self._per_cdr_regions(batch)
            if not regions:
                continue
 
            true_tokens = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
            seq_mask    = batch.get("seq_mask")
            coords_raw  = batch.get("coords")         # (B, L, 4, 3) or None
            native_raw  = batch.get("native_coords")  # ground-truth, optional
            if true_tokens is None or seq_mask is None:
                continue
 
            ag, ag_m, ag_s = (
                batch.get("antigen_coords"),
                batch.get("antigen_coord_mask"),
                batch.get("antigen_seq_tokens"),
            )
 
            conditions = {
                "full":  (ag,   ag_m,  ag_s),
                "no_ag": (None, None,  None),
            }
 
            for cond_name, (ac, acm, asts) in conditions.items():
                # ── teacher-forced forward pass ──────────────────────────
                batch["__model__"] = self.model
                logits = self._forward_ar_logits(batch, ac, acm, asts)
                batch.pop("__model__", None)
 
                if logits is None:
                    continue
                logits      = logits.float()           # (B, L-1, V)
                B, Lm1, _V  = logits.shape
                pred_tokens = logits.argmax(dim=-1)    # (B, L-1)
 
                for region_name, region_mask_full in regions.items():
                    # ── AAR ──────────────────────────────────────────────
                    # AR logits[:,p-1,:] predict token at position p,
                    # so shift the region mask by one to align.
                    r_mask  = region_mask_full[:, 1:].contiguous()[:, :Lm1]
                    ref_tok = true_tokens[:, 1:].contiguous()[:, :Lm1]
                    s_mask  = seq_mask[:, 1:].contiguous()[:, :Lm1]
 
                    valid = (
                        r_mask.bool()
                        & s_mask.bool()
                        & (ref_tok != _PAD_IDX)
                        & (ref_tok != 4)   # exclude MASK token
                    )
                    if valid.any():
                        acc[cond_name][region_name]["correct"] += int(
                            ((pred_tokens == ref_tok) & valid).sum()
                        )
                        acc[cond_name][region_name]["total"] += int(valid.sum())
 
                    # # ── Ca RMSD ──────────────────────────────────────────
                    # if native_raw is not None and coords_raw is not None:
                    #     sq, n = self._kabsch_rmsd_batch(
                    #         pred_coords   = coords_raw,
                    #         native_coords = native_raw,
                    #         region_mask   = region_mask_full,
                    #         seq_mask      = seq_mask,
                    #     )
                    #     acc[cond_name][region_name]["sq_err"] += sq
                    #     acc[cond_name][region_name]["n_res"]  += n
 
        # ── assemble flat metric dict ────────────────────────────────────
        metrics: Dict[str, float] = {}
        all_regions: set = set()
 
        for cond_name, region_acc in acc.items():
            for region_name, a in region_acc.items():
                all_regions.add(region_name)
                if a["total"] > 0:
                    metrics[f"aar_{region_name}_{cond_name}"] = (
                        a["correct"] / a["total"]
                    )
                if a["n_res"] > 0:
                    metrics[f"rmsd_{region_name}_{cond_name}"] = (
                        math.sqrt(a["sq_err"] / a["n_res"])
                    )
 
        # antigen-sensitivity gap: positive → model uses antigen identity
        for rn in all_regions:
            kf = f"aar_{rn}_full"
            kn = f"aar_{rn}_no_ag"
            if kf in metrics and kn in metrics:
                metrics[f"aar_{rn}_ag_gap"] = metrics[kf] - metrics[kn]
 
        if _was_training:
            self.model.train()
        return metrics
 
    # ── helpers used only by per_cdr_metrics ────────────────────────────
 
    def _per_cdr_regions(self, batch: dict) -> Dict[str, "torch.Tensor"]:
        """
        Build  { region_name: bool mask (B, L) }  from guaranteed batch keys:
            cdr3_mask, cdr_mask, chain_position_mask
 
        Optional key for individual H1/H2/L1/L2:
            cdr_region_ids  (B, L) int  -  codes: H1=0 H2=1 H3=2 L1=3 L2=4 L3=5
        """
        cdr3_raw  = batch.get("cdr3_mask")
        cdr_raw   = batch.get("cdr_mask")
        chain_raw = batch.get("chain_position_mask")
        if cdr3_raw is None or cdr_raw is None or chain_raw is None:
            return {}
 
        cdr3  = cdr3_raw.bool()
        cdr   = cdr_raw.bool()
        heavy = (chain_raw == 0)
        light = (chain_raw == 1)
 
        regions: Dict[str, "torch.Tensor"] = {
            "cdr3_heavy": cdr3 & heavy,   # H3  - exact Table 2 match
            "cdr3_light": cdr3 & light,   # L3  - exact Table 2 match
            "all_cdr":    cdr,            # H1+H2+H3+L1+L2+L3 pooled
            "all_cdr_h":  cdr & heavy,    # H1+H2+H3 pooled
            "all_cdr_l":  cdr & light,    # L1+L2+L3 pooled
        }
 
        return regions
 
    @staticmethod
    def _forward_ar_logits(
        batch: dict,
        antigen_coords,
        antigen_coord_mask,
        antigen_seq_tokens,
    ):
        """
        Single teacher-forced forward pass; returns AR logits (B, L-1, V)
        or None on error.
 
        The model is read from batch["__model__"] - a temporary slot set by
        per_cdr_metrics() so this method stays a @staticmethod (easily
        testable without instantiating the full class).
 
        Masking convention matches _batch_ar_nll: CDR positions are masked
        to token 4 in the decoder input so the model must predict them from
        context, exactly as during training.
        """
        model = batch.get("__model__")
        if model is None:
            return None
 
        clean  = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
        masked = clean.clone()
        cdr_b  = batch.get("cdr_mask")
        if cdr_b is not None:
            masked = masked.masked_fill(cdr_b.bool(), 4)
 
        try:
            out = model(
                seq_tokens         = masked,
                seq_tokens_orig    = masked,
                geom_seq_tokens    = clean,
                seq_mask           = batch.get("seq_mask"),
                cdr_mask           = batch.get("cdr_mask"),
                coords             = batch.get("coords"),
                coord_mask         = batch.get("coord_mask"),
                antigen_coords     = antigen_coords,
                antigen_coord_mask = antigen_coord_mask,
                antigen_seq_tokens = antigen_seq_tokens,
                task_mask          = {"AR": True, "MLM": False,
                                      "GEO": False, "IF": False},
                metadata           = batch.get("metadata"),
                metadata_indices   = batch.get("metadata_indices"),
                chain_mask         = batch.get("chain_position_mask"),
                cdr3_lengths       = batch.get("cdr3_lengths"),
                cdr3_mask          = batch.get("cdr3_mask"),
            )
        except Exception:
            return None
 
        return out.get("logits_ar")   # (B, L-1, V) or None

    # Private helpers
    def _filter_real_pairs(self, batch):
        if "synthetic_negative_mask" not in batch:
            return batch

        real_mask = ~batch["synthetic_negative_mask"]
        if real_mask.sum() == 0:
            return None

        filtered = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] == real_mask.shape[0]:
                filtered[k] = v[real_mask]
            elif isinstance(v, dict):
                filtered[k] = {
                    kk: vv[real_mask]
                    if torch.is_tensor(vv) and vv.shape[0] == real_mask.shape[0]
                    else vv
                    for kk, vv in v.items()
                }
            else:
                filtered[k] = v

        targets = batch.get("targets")
        if isinstance(targets, list):
            filtered["targets"] = [
                t for t, keep in zip(targets, real_mask.tolist()) if keep
            ]

        return filtered  # <- return new dict, don't mutate in-place
    
    
    def _batch_ar_nll(
        self,
        batch:         dict,
        antigen_coords,
        antigen_coord_mask,
        antigen_seq_tokens,
    ):
        """
        Single forward pass with the given antigen condition.
        Returns (total_nll, n_tokens, cdr3_nll, cdr3_n_tokens).
        Returns (None, 0, 0, 0) on error.
        """
        clean = batch.get("seq_tokens_orig", batch.get("seq_tokens"))
        masked_in = clean.clone() # type: ignore
        _cdr_b = batch.get("cdr_mask")
        
        if _cdr_b is not None:
            masked_in = masked_in.masked_fill(_cdr_b.bool(), 4)  # deny template
        
        # seq_tokens_orig must also be CDR-masked for the geom encoder path;
        # the model will re-apply the geom chemistry mask internally, but
        # passing clean orig here is consistent with training behaviour.
        # seq_tokens_orig_for_geom = clean  # geom mask applied inside model forward
        
        try:
            outputs = self.model(
                seq_tokens         = masked_in,   # CDR-masked decoder input
                seq_tokens_orig    = masked_in,
                # seq_tokens_orig    = clean,
                geom_seq_tokens    = clean,
                seq_mask           = batch.get("seq_mask"),
                cdr_mask           = batch.get("cdr_mask"),
                coords             = batch.get("coords"),
                coord_mask         = batch.get("coord_mask"),
                antigen_coords     = antigen_coords,
                antigen_coord_mask = antigen_coord_mask,
                antigen_seq_tokens = antigen_seq_tokens,
                task_mask          = {"AR": True, "MLM": False, "GEO": False, "IF": False},
                metadata           = batch.get("metadata"),
                metadata_indices   = batch.get("metadata_indices"),
                chain_mask         = batch.get("chain_position_mask"),
                cdr3_lengths       = batch.get("cdr3_lengths"),
                cdr3_mask          = batch.get("cdr3_mask"),
            )
        except Exception:
            return None, 0, 0.0, 0

        if "logits_ar" not in outputs:
            return None, 0, 0.0, 0

        logits   = outputs["logits_ar"].float()        # (B, L-1, V)
        valid_rows = batch["sample_task_mask"]["AR"]
        logits = logits[valid_rows]
        
        _clean   = batch.get("seq_tokens_orig", batch["seq_tokens"])
        tokens   = _clean[:, 1:].contiguous()   # (B, L-1)
        seq_mask = batch["seq_mask"][:, 1:].contiguous()     # (B, L-1)

        min_len = min(logits.shape[1], tokens.shape[1])
        logits  = logits[:, :min_len]
        tokens = _clean[valid_rows][:, 1:].contiguous()
        seq_mask = batch["seq_mask"][valid_rows][:, 1:].contiguous()

        # Exclude PAD and MASK tokens from loss
        valid = seq_mask.bool() & (tokens != _PAD_IDX) & (tokens != 4)

        nll_per_tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
            ignore_index=_IGNORE_IDX,
        ).reshape(tokens.shape)    # (B, L-1)

        total_nll = nll_per_tok[valid].sum().item()
        n_tokens  = int(valid.sum().item())

        # CDR3 only
        cdr3_nll, cdr3_n = 0.0, 0
        cdr3_mask_raw = batch.get("cdr3_mask")
        if cdr3_mask_raw is not None:
            cdr3_mask_raw = cdr3_mask_raw[valid_rows]
            cdr3_m = cdr3_mask_raw[:, 1:].contiguous()[:, :min_len]
            cdr3_valid = valid & cdr3_m.bool()
            if cdr3_valid.any():
                cdr3_nll = nll_per_tok[cdr3_valid].sum().item()
                cdr3_n   = int(cdr3_valid.sum().item())

        return total_nll, n_tokens, cdr3_nll, cdr3_n

    # Ablation transformations 
    @staticmethod
    def _cross_class_perm(B, targets):
        """
        For each sample i, choose j from a different target class.

        Returns:
            List[int] permutation indices.
        """
        if (
            targets is None
            or len({str(t) for t in targets if t is not None}) < 2
        ):
            return torch.roll(
                torch.arange(B),
                shifts=max(1, B // 2)
            ).tolist()

        import random

        idx = list(range(B))
        for i in range(B):
            candidates = [
                j for j in range(B)
                if targets[j] is not None
                and targets[j] != targets[i]
            ]

            if candidates:
                idx[i] = random.choice(candidates)

        return idx


    @staticmethod
    def _apply_perm(tensor, perm):
        if tensor is None:
            return None
        return tensor[perm]

    @staticmethod
    def _randomize(coords: torch.Tensor) -> torch.Tensor:
        """
        Replace antigen coordinates with isotropic Gaussian noise
        (mean=0, std=5 A, matching typical protein coordinate scale).
        The noise is generated fresh each call so the random_ag condition
        averages over many noise realizations across batches.
        """
        return torch.randn_like(coords) * 5.0

    # Metric assembly 
    @staticmethod
    def _ppl(nll_sum: float, n_tokens: int) -> Optional[float]:
        if n_tokens == 0:
            return None
        return math.exp(nll_sum / n_tokens)

    def _build_metrics(self, accum: dict) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        ppl = {}
        cdr3_ppl = {}
        for cond, a in accum.items():
            p = self._ppl(a["nll_sum"],      a["n_tokens"])
            c = self._ppl(a["cdr3_nll_sum"], a["cdr3_tokens"])
            if p is not None:
                ppl[cond]      = p
                metrics[f"ar_ppl_{cond}"] = p
            if c is not None:
                cdr3_ppl[cond] = c
                metrics[f"cdr3_ppl_{cond}"] = c

        # Gap metrics - only meaningful when full baseline is available
        if "full" in ppl:
            for cond in ("no_ag", "shuffled_ag", "random_ag"):
                if cond in ppl:
                    metrics[f"ppl_gap_{cond}"] = ppl[cond] - ppl["full"]
                if cond in cdr3_ppl and "full" in cdr3_ppl:
                    metrics[f"cdr3_ppl_gap_{cond}"] = cdr3_ppl[cond] - cdr3_ppl["full"]

        # antigen_presence_active: model uses antigen being present vs absent.
        # Activates before specificity - expect this first in early Phase 2.
        gap_no_ag = metrics.get("ppl_gap_no_ag", 0.0)
        metrics["antigen_presence_active"] = float(gap_no_ag > 0.02)
        
        # antigen_specificity_active: model uses antigen *identity*, not just presence.
        # True when shuffled antigen (wrong identity) is worse than real antigen.
        # Threshold 0.005: conservative but achievable; 0.01 was set before the
        # shift-by-1 bias was understood (adjacent samples often share antigen class).
        gap_shuffled     = metrics.get("ppl_gap_shuffled_ag", 0.0)
        cdr3_gap_shuffled = metrics.get("cdr3_ppl_gap_shuffled_ag", 0.0)

        metrics["antigen_specificity_active"]      = float(gap_shuffled > 0.005)
        metrics["antigen_specificity_score"]       = gap_shuffled        # continuous
        metrics["cdr3_antigen_specificity_active"] = float(cdr3_gap_shuffled > 0.003)
        metrics["cdr3_antigen_specificity_score"]  = cdr3_gap_shuffled   # continuous

        return metrics
