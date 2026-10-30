# model/latent_probe.py
"""
Latent Space Probe for SCAT.

"""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

#optional deps - fail gracefully if not available ##
try:
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    silhouette_score = None
    PCA = None
    print("[LatentProbe] scikit-learn not found - silhouette/PCA disabled")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False



# Hook-based representation extractor
class _HookStore:
    """Minimal forward-hook accumulator."""

    def __init__(self):
        self._store: Dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, name: str, module: torch.nn.Module, take_input: bool = False):
        def _hook(mod, inp, out):
            if take_input:
                self._store[name] = (
                    inp[0].detach().cpu() if isinstance(inp, tuple)
                    else inp.detach().cpu()
                )
            else:
                if isinstance(out, tuple):
                    out = out[0]
                self._store[name] = out.detach().cpu()
        self._handles.append(module.register_forward_hook(_hook))

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._store.get(name)

    def clear(self):
        self._store.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# Core probe
class LatentProbe:
    """
    Attaches to a SCATModel via forward hooks and periodically snapshots
    the latent geometry across a validation loader.

    Usage:
        probe = LatentProbe(model, cfg, snapshot_dir="latent_snapshots/")
        # call at every validation checkpoint:
        probe.run(val_loaders, global_step, device)
    """

    def __init__(self, model, cfg: Dict, snapshot_dir: str = "latent_snapshots"):
        self.model = model
        self.cfg   = cfg
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.max_batches = cfg.get("latent_probe", {}).get("max_batches", 50)
        self.run_umap    = cfg.get("latent_probe", {}).get("run_umap", False) and HAS_UMAP
        self.enabled     = cfg.get("latent_probe", {}).get("enabled", True)

        # warn when max_batches is unusually high
        if self.max_batches > 100:
            print(
                f"[LatentProbe] max_batches={self.max_batches} is high. "
                f"Consider setting it to ≤100 to keep probe time under ~60s."
            )

        self._hooks = None

        # History for metric trending
        self.history: Dict[str, List] = defaultdict(list)

    # hook attachment
    def _attach_hooks(self):
        if self._hooks is None:
            self._hooks = _HookStore()
        m = self.model
        # Unwrap DDP if needed
        if hasattr(m, "module"):
            m = m.module

        self._hooks.register("seq_enc",   m.seq_encoder)

    def detach(self):
        if self._hooks is not None:
            self._hooks.remove()
            self._hooks = None

    # Main entry
    @torch.no_grad()
    def run(
        self,
        val_loaders: Dict[str, Any],
        global_step: int,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run probing pass and return scalar metrics for logging.
        Also writes a .npz snapshot to disk.
        """
        if not self.enabled:
            return {}

        self.model.eval()
        t0 = time.time()
        
        self._hooks = _HookStore()
        self._attach_hooks()
        
        try:
            collector = _LatentCollector()

            for ds_name, loader in val_loaders.items():
                self._collect_from_loader(loader, ds_name, device, collector)

            metrics = self._compute_metrics(collector, global_step)

            if collector.n_samples > 0:
                self._save_snapshot(collector, global_step)
            else:
                print("[LatentProbe] no samples collected - snapshot skipped")

            self._update_history(metrics, global_step)

            elapsed = time.time() - t0
            print(
                f"[LatentProbe] step={global_step}  elapsed={elapsed:.1f}s  "
                f"n_samples={collector.n_samples}"
            )

            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            return metrics

        finally:
            if self._hooks is not None:
                self._hooks.remove()
                self._hooks = None
    
    def _pairwise_cosine_dist(self, x: np.ndarray, max_samples: int = 300) -> np.ndarray:
        if x.shape[0] > max_samples:
            idx = np.random.choice(x.shape[0], max_samples, replace=False)
            x = x[idx]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = x @ x.T
        return 1.0 - sim  # cosine distance
    
    @staticmethod
    def _nn_retrieval(src_arr, tgt_arr, ks=(1, 5), max_n=500):
        """Fraction of samples whose top-k src-NN overlap with top-k tgt-NN."""
        n = min(max_n, src_arr.shape[0])
        if n < 20:
            return {}
        idx = np.random.choice(src_arr.shape[0], n, replace=False)
        s = src_arr[idx]
        t = tgt_arr[idx]
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
        sim_s = s @ s.T
        sim_t = t @ t.T
        np.fill_diagonal(sim_s, -np.inf)
        np.fill_diagonal(sim_t, -np.inf)
        result = {}
        for k in ks:
            nn_s = np.argsort(-sim_s, axis=1)[:, :k]
            nn_t = np.argsort(-sim_t, axis=1)[:, :k]
            hits = [len(set(nn_s[i]) & set(nn_t[i])) > 0 for i in range(n)]
            result[k] = float(np.mean(hits))
        return result

    # collection
    def _collect_from_loader(self, loader, ds_name: str, device, collector):
        
        if self._hooks is None:
            raise RuntimeError(
                "[LatentProbe] hooks not attached. "
                "run() must be used to collect latents."
            )
            
        for i, batch in enumerate(loader):
            if i >= self.max_batches:
                break

            if batch is None:
                continue

            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            self._hooks.clear()

            # Forward pass with all tasks active so all encoders fire
            task_mask = {"AR": True, "MLM": False, "GEO": True, "IF": True}
            try:
                _outputs = self.model(
                    seq_tokens        = batch.get("seq_tokens"),
                    seq_tokens_orig   = batch.get("seq_tokens_orig"),
                    seq_mask          = batch.get("seq_mask"),
                    cdr_mask          = batch.get("cdr_mask"),
                    mlm_labels        = batch.get("mlm_labels"),
                    coords            = batch.get("coords"),
                    coord_mask        = batch.get("coord_mask"),
                    antigen_coords    = batch.get("antigen_coords"),
                    antigen_coord_mask= batch.get("antigen_coord_mask"),
                    interface_labels  = batch.get("interface_labels"),
                    interface_mask    = batch.get("interface_mask"),
                    task_mask         = task_mask,
                )
            except Exception as e:
                print(f"[LatentProbe] forward error in {ds_name} batch {i}: {e}")
                continue

            # Extract representations from hooks
            seq_enc_t  = self._hooks.get("seq_enc")     # (B, L, D) or None
            _cond_repr = _outputs.get("cond_repr")       # [B, D] already squeezed
            cond_t     = _cond_repr.detach().cpu() if _cond_repr is not None else None
            _geom_raw = _outputs.get("geom_repr")
            geom_t = _geom_raw.detach().cpu() if _geom_raw is not None else None
            _ag_raw = _outputs.get("antigen_pooled")
            antigen_t = _ag_raw.detach().cpu() if _ag_raw is not None else None
            
            real_mask = ~batch["synthetic_negative_mask"]

            if seq_enc_t is None or seq_enc_t.dim() != 3:
                print(
                    f"[LatentProbe] skipping batch {i} in {ds_name} - "
                    f"seq_enc has unexpected shape: "
                    f"{seq_enc_t.shape if seq_enc_t is not None else 'None'}"
                )
                continue

            # real_mask_dev = real_mask.to(seq_enc_t.device)
            real_mask_cpu = real_mask.cpu()
            real_mask_dev = real_mask.to(seq_enc_t.device)

            if cond_t is not None:
                cond_t = cond_t[real_mask_cpu] # was real_mask_dev

            if geom_t is not None:
                geom_t = geom_t[real_mask_cpu]

            if antigen_t is not None:
                antigen_t = antigen_t[real_mask_cpu]

            # Skip fully synthetic batches
            if real_mask.sum() == 0:
                continue

            B = int(real_mask.sum().item())

            seq_enc_t = seq_enc_t[real_mask_cpu]
            seq_mask_cpu = batch["seq_mask"][real_mask].cpu()

            # CDR mask resolution (three-tier fallback)
            # Tier 1: OAS - exact masks from mmap (cdr_mask = CDR1+2+3)
            # Tier 2: SAbDab - derive approx from heavy_len / light_len
            # Tier 3: no CDR info - uniform pooling, CDR3 metrics skipped
            full_cdr_mask_cpu = batch.get("cdr_mask")
            if full_cdr_mask_cpu is not None:
                full_cdr_mask_cpu = full_cdr_mask_cpu[real_mask].cpu().bool()
            
            cdr3_mask_cpu = batch.get("cdr3_mask")
            if cdr3_mask_cpu is not None:
                cdr3_mask_cpu = cdr3_mask_cpu[real_mask].cpu().bool()

            # Tier-2 fallback: approximate from heavy_len / light_len (SAbDab)
            if full_cdr_mask_cpu is None:
                heavy_lens = batch.get("heavy_len")
                light_lens = batch.get("light_len")

                if heavy_lens is not None:
                    heavy_lens = heavy_lens[real_mask]

                if light_lens is not None:
                    light_lens = light_lens[real_mask]
                    
                if heavy_lens is not None:
                    full_cdr_mask_cpu, cdr3_mask_cpu = self._derive_cdr_masks_approx(
                        B, seq_enc_t.shape[1], heavy_lens, light_lens
                    )

            # Weighted pool: CDR positions boosted 3× over framework
            seq_pooled = self._weighted_pool(
                seq_enc_t, seq_mask_cpu, full_cdr_mask_cpu
            )  # (B, D)
            
            _sp_raw = _outputs.get("seq_pooled")
            if _sp_raw is not None:
                seq_pooled = _sp_raw.detach().cpu()[real_mask.cpu()]

            # CDR3 pool: exact or approximate; None -> metric skipped
            cdr3_pooled = self._cdr3_pool(seq_enc_t, cdr3_mask_cpu)  # (B, D) or None

            # assert cond shape after squeeze
            cond_pooled = None
            if cond_t is not None:
                cond_pooled = cond_t.squeeze(1) if cond_t.dim() == 3 else cond_t
                assert cond_pooled.dim() == 2, (
                    f"[LatentProbe] cond should be (B, D) after squeeze, "
                    f"got {cond_t.shape}"
                )

            # separate cond buffers aligned to geom / antigen
            # cond_for_geom is only populated when geom_t is also present
            # in this batch, ensuring shapes always match for alignment metric.
            cond_for_antigen = cond_pooled.numpy() if (cond_pooled is not None and antigen_t is not None) else None
            cond_for_geom    = None
            seq_for_geom     = None

            cdr3_lengths = self._estimate_cdr3_length(cdr3_mask_cpu, seq_mask_cpu)

            collector.add(
                dataset          = ds_name,
                seq_pooled       = seq_pooled.numpy(),
                cond             = cond_pooled.numpy() if cond_pooled is not None else None,
                cond_for_geom    = cond_for_geom,
                seq_for_geom     = seq_for_geom,
                cond_for_antigen = cond_for_antigen,
                geom             = geom_t.numpy() if geom_t is not None else None,
                antigen          = antigen_t.numpy() if antigen_t is not None else None,
                cdr3_pool        = cdr3_pooled.numpy() if cdr3_pooled is not None else None,
                cdr3_lens        = cdr3_lengths,
                has_struct       = [batch.get("coords") is not None] * B,
                has_antigen      = [batch.get("antigen_coords") is not None] * B,
            )

    # metrics
    def _compute_metrics(self, collector, global_step: int) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if collector.n_samples < 10:
            print(
                f"[LatentProbe] only {collector.n_samples} samples - skipping metrics"
            )
            return metrics

        seq_arr   = collector.array("seq_pooled")   # (N, D)
        cond_arr  = collector.array("cond")          # (N, D) or None
        labels    = np.array(collector.all_datasets) # (N,)
        cdr3_lens = np.array(collector.all_cdr3_lens)

        # use shape-aligned buffers for alignment metrics
        geom_arr         = collector.array("geom")
        antigen_arr      = collector.array("antigen")
        cond_for_antigen = collector.array("cond_for_antigen")

        # 1. Antigen–cond alignment
        if antigen_arr is not None and cond_for_antigen is not None:
            if antigen_arr.shape[0] == cond_for_antigen.shape[0] and antigen_arr.shape[0] > 0:
                a = antigen_arr      / (np.linalg.norm(antigen_arr,      axis=1, keepdims=True) + 1e-8)
                c = cond_for_antigen / (np.linalg.norm(cond_for_antigen, axis=1, keepdims=True) + 1e-8)
                cos_vals = np.sum(a * c, axis=1)
                metrics["antigen_cond_alignment"] = float(cos_vals.mean())
                metrics["antigen_cond_alignment_std"] = float(cos_vals.std())
            else:
                print(
                    f"[LatentProbe] antigen alignment skipped: "
                    f"antigen={antigen_arr.shape[0]} cond_for_antigen={cond_for_antigen.shape[0]}"
                )

        # 2. Dataset silhouette in seq_pooled
        # Should be LOW - the model should NOT memorize dataset identity.
        # if HAS_SKLEARN and len(np.unique(labels)) > 1 and len(seq_arr) > 20:
        if (
            HAS_SKLEARN
            and silhouette_score is not None
            and PCA is not None
            and len(np.unique(labels)) > 1
            and len(seq_arr) > 20
        ):
            try:
                n_comp = min(50, seq_arr.shape[0] - 1, seq_arr.shape[1])
                pca = PCA(n_components=n_comp)
                seq_pca = pca.fit_transform(seq_arr)
                sil = silhouette_score(
                    seq_pca, labels, metric="cosine",
                    sample_size=min(2000, len(seq_arr))
                )
                metrics["dataset_silhouette_seq"] = float(sil)
                metrics["pca_var_50d"] = float(pca.explained_variance_ratio_.sum())
            except Exception as e:
                print(f"[LatentProbe] silhouette error: {e}")
        
        # 2b. Dataset silhouette in cond space
        # Should also be LOW - cond should not cluster by dataset
        if (HAS_SKLEARN and silhouette_score is not None and PCA is not None
            and cond_arr is not None
            and len(np.unique(labels)) > 1
            and len(cond_arr) > 20):
            try:
                n_comp = min(50, cond_arr.shape[0] - 1, cond_arr.shape[1])
                pca_c = PCA(n_components=n_comp)
                cond_pca = pca_c.fit_transform(cond_arr)
                sil_cond = silhouette_score(
                    cond_pca, labels, metric="cosine",
                    sample_size=min(2000, len(cond_arr))
                )
                metrics["dataset_silhouette_cond"] = float(sil_cond)
            except Exception as e:
                print(f"[LatentProbe] cond silhouette error: {e}")

        # 3. CDR3 length variance explained by PC1
        # Should RISE - the first PC of seq_pooled should increasingly
        # correlate with CDR3 length as the model learns it is the primary
        # axis of antibody diversity.
        if HAS_SKLEARN and len(seq_arr) > 20 and cdr3_lens is not None and PCA is not None:
            valid = cdr3_lens > 0
            if valid.sum() > 10:
                try:
                    lens_valid = cdr3_lens[valid]
                    # Skip when all CDR3 lengths are identical (zero variance)
                    # - np.corrcoef would produce NaN and RuntimeWarning
                    if lens_valid.std() > 0:
                        pca2 = PCA(n_components=1)
                        pc1  = pca2.fit_transform(seq_arr[valid]).ravel()
                        corr = np.corrcoef(pc1, lens_valid)[0, 1]
                        if np.isfinite(corr):
                            metrics["cdr3_len_pc1_corr"] = float(abs(corr))
                except Exception as e:
                    print(f"[LatentProbe] CDR3 corr error: {e}")

        # 4. SAbDab conditioning cohesion
        # For SAbDab batches: how tightly do the cond vectors cluster?
        # Should increase as the model learns consistent structural conditioning.
        if cond_arr is not None:
            sabdab_mask = labels == "sabdab"
            if sabdab_mask.sum() > 5:
                sb_cond   = cond_arr[sabdab_mask]
                sb_cond_n = sb_cond / (np.linalg.norm(sb_cond, axis=1, keepdims=True) + 1e-8)
                mean_cond = sb_cond_n.mean(axis=0)
                cosines   = sb_cond_n @ mean_cond
                metrics["sabdab_cond_cohesion"] = float(cosines.mean())

        # 5. Representation norm health (extended)
        norms = np.linalg.norm(seq_arr, axis=1)
        metrics["seq_pooled_norm_mean"] = float(norms.mean())
        metrics["seq_pooled_norm_std"]  = float(norms.std())

        if cond_arr is not None:
            c_norms = np.linalg.norm(cond_arr, axis=1)
            metrics["cond_norm_mean"] = float(c_norms.mean())
            metrics["cond_norm_std"]  = float(c_norms.std())

        if geom_arr is not None:
            g_norms = np.linalg.norm(geom_arr, axis=1)
            metrics["geom_norm_mean"] = float(g_norms.mean())
            metrics["geom_norm_std"]  = float(g_norms.std())

        if antigen_arr is not None:
            a_norms = np.linalg.norm(antigen_arr, axis=1)
            metrics["antigen_norm_mean"] = float(a_norms.mean())
            metrics["antigen_norm_std"]  = float(a_norms.std())

        # 6. Pairwise distance scale (detect collapse vs mixing)
        try:
            if seq_arr is not None and len(seq_arr) > 10:
                n = min(300, seq_arr.shape[0])  # subsample
                idx = np.random.choice(seq_arr.shape[0], n, replace=False)
                d = self._pairwise_cosine_dist(seq_arr[idx])
                iu = np.triu_indices_from(d, k=1)
                d_flat = d[iu]

                metrics["seq_pairwise_dist_mean"] = float(d_flat.mean())
                metrics["seq_pairwise_dist_std"]  = float(d_flat.std())
        except Exception as e:
            print(f"[LatentProbe] pairwise dist error: {e}")
            
        # 7. Nearest-neighbour retrieval

        if antigen_arr is not None and cond_for_antigen is not None:
            try:
                ret = self._nn_retrieval(antigen_arr, cond_for_antigen)
                metrics["antigen_nn_retrieval_at1"] = ret.get(1, float("nan"))
                metrics["antigen_nn_retrieval_at5"] = ret.get(5, float("nan"))
            except Exception as e:
                print(f"[LatentProbe] antigen retrieval error: {e}")
                
        # 8. Conditioning effective rank
        # Tracks whether cond is collapsing to a low-dimensional subspace.
        # effective_rank = exp(entropy of variance spectrum)
        # Full rank (256) -> ~256
        # Collapsed representation -> approaches 1

        if cond_arr is not None and len(cond_arr) > 20:
            try:
                n = min(500, cond_arr.shape[0])
                idx = np.random.choice(cond_arr.shape[0], n, replace=False)

                sub = cond_arr[idx].astype(np.float32)

                # center features before SVD
                sub_c = sub - sub.mean(axis=0, keepdims=True)

                _, sv, _ = np.linalg.svd(sub_c, full_matrices=False)

                # variance spectrum
                sv2 = sv ** 2
                sv2 = sv2 / (sv2.sum() + 1e-10)

                eff_rank = float(
                    np.exp(
                        -np.sum(sv2 * np.log(sv2 + 1e-10))
                    )
                )

                metrics["cond_effective_rank"] = eff_rank

            except Exception as e:
                print(f"[LatentProbe] effective rank error: {e}")
        
        # 9. Per-dataset effective rank
        for ds_name in ['sabdab', 'oas_hl', 'oas_paired']:
            mask = labels == ds_name
            if mask.sum() > 20:
                try:
                    sub = seq_arr[mask][:min(300, mask.sum())].astype(np.float32)
                    sub_c = sub - sub.mean(axis=0)
                    _, sv, _ = np.linalg.svd(sub_c, full_matrices=False)
                    sv2 = sv**2; sv2 /= sv2.sum() + 1e-10
                    pr = float(1.0 / (sv2**2).sum())
                    er = float(np.exp(-np.sum(sv2 * np.log(sv2 + 1e-10))))
                    metrics[f"seq_eff_rank_{ds_name}"] = er
                    metrics[f"seq_pr_{ds_name}"] = pr
                except Exception as e:
                    print(f"[LatentProbe] per-dataset rank error ({ds_name}): {e}")
        

        return metrics

    # snapshot
    def _save_snapshot(self, collector, global_step: int):
        """
        Save raw latent arrays as .npz for offline analysis / UMAP later.
        Each snapshot is self-contained with all metadata needed for plotting.
        """
        save_dict: Dict[str, Any] = {
            "global_step": global_step,
            "n_samples": collector.n_samples,
            "datasets":    np.array(collector.all_datasets),
            "cdr3_lens":   np.array(collector.all_cdr3_lens),
            "has_struct":  np.array(collector.all_has_struct),
            "has_antigen": np.array(collector.all_has_antigen),
        }

        for name in ["seq_pooled", "cond", "geom", "antigen", "cdr3_pool",
                     "cond_for_geom", "cond_for_antigen"]:
            arr = collector.array(name)
            if arr is not None:
                save_dict[name] = arr

        fname = self.snapshot_dir / f"latent_step_{global_step:07d}.npz"
        np.savez_compressed(fname, **save_dict)
        print(f"[LatentProbe] Snapshot saved -> {fname}")

        # Optional UMAP (expensive - off by default)
        if self.run_umap and "seq_pooled" in save_dict:
            self._save_umap(save_dict, global_step)

    def _save_umap(self, save_dict: Dict, global_step: int):
        try:
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=30,
                min_dist=0.1, metric="cosine", low_memory=True, verbose=False
            )
            emb = reducer.fit_transform(save_dict["seq_pooled"])
            umap_path = self.snapshot_dir / f"umap_step_{global_step:07d}.npz"
            np.savez_compressed(
                umap_path,
                embedding  = emb,
                datasets   = save_dict["datasets"],
                cdr3_lens  = save_dict["cdr3_lens"],
            )
            print(f"[LatentProbe] UMAP saved -> {umap_path}")
        except Exception as e:
            print(f"[LatentProbe] UMAP failed: {e}")

    def _update_history(self, metrics: Dict[str, float], step: int):
        self.history["step"].append(step)
        for k, v in metrics.items():
            self.history[k].append(v)

    def save_history(self, path: str = "latent_history.npz"):
        np.savez_compressed(
            path, **{k: np.array(v) for k, v in self.history.items()}
        )

    # static helpers
    @staticmethod
    def _derive_cdr_masks_approx(
        B: int,
        L: int,
        heavy_lens,
        light_lens,
    ) -> "Tuple[torch.Tensor, torch.Tensor]":
        """
        Approximate CDR masks for datasets with no explicit CDR annotation
        (e.g. SAbDab). Uses Chothia scheme absolute positions, stable for
        typical human VH (~117-125 aa) and VL (~107-113 aa) lengths.

        Chothia 0-based ranges:
          VH: CDR1 [25:35], CDR2 [51:56], CDR3 [94:105]
          VL: CDR1 [25:32], CDR2 [49:52], CDR3 [90:97]
        """
        VH_CDR_SPANS = [(25, 35), (51, 56), (94, 105)]
        VH_CDR3_SPAN = (94, 105)
        VL_CDR_SPANS = [(25, 32), (49, 52), (90, 97)]
        VL_CDR3_SPAN = (90, 97)

        full_mask = torch.zeros(B, L, dtype=torch.bool)
        cdr3_mask = torch.zeros(B, L, dtype=torch.bool)

        def _to_list(x, b):
            if x is None:
                return [0] * b
            if torch.is_tensor(x):
                return x.tolist()
            return [int(x)] * b

        h_lens = _to_list(heavy_lens, B)
        l_lens = _to_list(light_lens, B)

        for b in range(B):
            hl = int(h_lens[b])
            ll = int(l_lens[b])

            for s, e in VH_CDR_SPANS:
                cs, ce = min(s, hl), min(e, hl)
                if cs < ce:
                    full_mask[b, cs:ce] = True

            s3, e3 = VH_CDR3_SPAN
            cs3, ce3 = min(s3, hl), min(e3, hl)
            if cs3 < ce3:
                cdr3_mask[b, cs3:ce3] = True

            if ll > 0:
                offset = hl
                for s, e in VL_CDR_SPANS:
                    cs, ce = offset + min(s, ll), offset + min(e, ll)
                    if cs < ce and ce <= L:
                        full_mask[b, cs:ce] = True
                s3, e3 = VL_CDR3_SPAN
                cs3, ce3 = offset + min(s3, ll), offset + min(e3, ll)
                if cs3 < ce3 and ce3 <= L:
                    cdr3_mask[b, cs3:ce3] = True

        return full_mask, cdr3_mask

    @staticmethod
    def _weighted_pool(
        seq_enc: torch.Tensor,
        seq_mask: torch.Tensor,
        full_cdr_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CDR-boosted mean pooling over sequence encoder output."""
        # assert CPU to catch any future GPU migration bugs early
        assert seq_enc.device.type == "cpu", (
            f"[LatentProbe] seq_enc must be on CPU for probe pooling, "
            f"got device={seq_enc.device}"
        )
        enc = seq_enc.masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        if full_cdr_mask is not None:
            w = (1.0 + full_cdr_mask.float() * 2.0) * seq_mask.float()
        else:
            w = seq_mask.float()
        w = w.unsqueeze(-1)
        pooled = (enc * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        return pooled  # (B, D)

    @staticmethod
    def _cdr3_pool(
        seq_enc: torch.Tensor,
        cdr3_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Mean-pool CDR3 positions using the exact cdr3_mask from the dataset.
        No heuristic - cdr3_mask already encodes precise CDR3 boundaries.
        """
        if cdr3_mask is None:
            return None
        B, L, D = seq_enc.shape
        results = []
        for b in range(B):
            cm = cdr3_mask[b]           # (L,) bool
            if not cm.any():
                results.append(torch.zeros(D))
                continue
            cdr3_enc = seq_enc[b][cm]   # (cdr3_len, D)
            results.append(cdr3_enc.mean(dim=0))
        return torch.stack(results)     # (B, D)

    @staticmethod
    def _estimate_cdr3_length(
        cdr3_mask: Optional[torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> List[int]:
        """
        Return per-sample CDR3 length by summing the exact cdr3_mask.
        Returns zeros for samples with no CDR3 annotation.
        """
        if cdr3_mask is None:
            return [0] * seq_mask.shape[0]
        return cdr3_mask.sum(dim=1).tolist()  # (B,)

# Collector
class _LatentCollector:
    """Accumulates per-batch latents into numpy arrays."""

    def __init__(self):
        self._buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.all_datasets:   List[str]  = []
        self.all_cdr3_lens:  List[int]  = []
        self.all_has_struct:  List[bool] = []
        self.all_has_antigen: List[bool] = []
        self.n_samples = 0

    def add(
        self,
        dataset:          str,
        seq_pooled:       np.ndarray,
        cond:             Optional[np.ndarray],
        geom:             Optional[np.ndarray],
        antigen:          Optional[np.ndarray],
        cdr3_pool:        Optional[np.ndarray],
        cdr3_lens:        List[int],
        has_struct:       List[bool],
        has_antigen:      List[bool],
        cond_for_geom:    Optional[np.ndarray] = None,
        seq_for_geom:     Optional[np.ndarray] = None,
        cond_for_antigen: Optional[np.ndarray] = None,
    ):
        B = seq_pooled.shape[0]
        self._buffers["seq_pooled"].append(seq_pooled)

        if cond             is not None: self._buffers["cond"].append(cond)
        if geom             is not None: self._buffers["geom"].append(geom)
        if antigen          is not None: self._buffers["antigen"].append(antigen)
        if cdr3_pool        is not None: self._buffers["cdr3_pool"].append(cdr3_pool)
        if cond_for_geom    is not None: self._buffers["cond_for_geom"].append(cond_for_geom)
        if seq_for_geom     is not None: self._buffers["seq_for_geom"].append(seq_for_geom)
        if cond_for_antigen is not None: self._buffers["cond_for_antigen"].append(cond_for_antigen)

        self.all_datasets    += [dataset] * B
        self.all_cdr3_lens   += cdr3_lens
        self.all_has_struct  += has_struct
        self.all_has_antigen += has_antigen
        self.n_samples       += B

    def array(self, name: str) -> Optional[np.ndarray]:
        bufs = self._buffers.get(name)
        if not bufs:
            return None
        return np.concatenate(bufs, axis=0)