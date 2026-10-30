# train/validator.py
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logging import get_logger
import numpy as np
from utils.geometry import (
    compute_aligned_rmsd,
    compute_unaligned_rmsd, 
    compute_ca_ca_distances
)

logger = get_logger(__name__)


def is_main_process():
    """Check if this is the main process in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

@torch.no_grad()
def validate(model, dataloader, device, vocab_size=25, max_batches=None):
    """
    Validation for a single dataset.
    """
    model.eval()
    
    # For normalization
    import math
    max_ce = math.log(vocab_size)

    metrics_accum = {
        "ar_loss": [],
        "ar_token_nlls": [],
        "fw_correct": 0,
        "fw_total": 0,
        "cdr_correct": 0,
        "cdr_total": 0,
        "cdr3_correct": 0,
        "cdr3_total": 0,
        "cdr_only_nlls": [],
        "cdr3_only_nlls": [],
        "h3_only_nlls": [],
        "l3_only_nlls": [],
        "mlm_loss": [],
        "mlm_correct": 0,
        "mlm_total": 0,
        "mlm_token_nlls": [],
        "geo_rmsd_aligned": [],      
        "geo_rmsd_unaligned": [],    
        "geo_ca_ca_mean": [],        
        "geo_ca_ca_std": [],
        "geo_ca_ca_deviation": [],
        "iface_logits": [], 
        "iface_labels": [],
        "residue_iface_logits": [],
        "residue_iface_labels": [],
        "cdr3_per_sample_accs": [],
        "heavy_cdr3_correct": 0,
        "heavy_cdr3_total": 0,
        "light_cdr3_correct": 0,
        "light_cdr3_total": 0,
    }

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        
        if batch is None:
            continue

        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(device)

        # Forward pass
        try:
            outputs = model(
                seq_tokens=batch.get("seq_tokens"),
                seq_mask=batch.get("seq_mask"),
                cdr_mask=batch.get("cdr_mask"),
                mlm_labels=batch.get("mlm_labels"),
                coords=batch.get("coords"),
                coord_mask=batch.get("coord_mask"),
                antigen_coords=batch.get("antigen_coords"),
                antigen_coord_mask=batch.get("antigen_coord_mask"),
                interface_labels=batch.get("interface_labels"),
                interface_mask=batch.get("interface_mask"),
                task_mask=batch.get("task_mask"),
                metadata=batch.get("metadata"),
                metadata_indices=batch.get("metadata_indices"),
                chain_mask = batch.get("chain_position_mask"),
                cdr3_lengths = batch.get("cdr3_lengths"),
                antigen_seq_tokens = batch.get("antigen_seq_tokens"),
            )
        except Exception as e:
            if is_main_process():
                logger.error(f"Error in validation forward pass (batch {batch_idx}): {e}")
                import traceback
                logger.error(traceback.format_exc())
            continue

        task_mask = batch.get("task_mask", {})
        real_mask = ~batch["synthetic_negative_mask"]
        
        if real_mask.sum() == 0:
            continue
        
        # ----- Clean AR pass -----
        # The main forward uses batch["seq_tokens"], which collate corrupts in place
        # with MLM MASK (clean copy = seq_tokens_orig), and it omits cdr3_mask so the
        # positional signal is absent. Both depress AR/CDR3. Re-run AR alone on
        # clean tokens WITH cdr3_mask so the metrics reflect true model behaviour.
        ar_outputs = outputs
        ar_tokens  = batch.get("seq_tokens")
        if task_mask.get("AR", False):
            clean_tokens = batch.get("seq_tokens_orig")
            if clean_tokens is None:
                clean_tokens = batch.get("seq_tokens")
            try:
                ar_outputs = model(
                    seq_tokens         = clean_tokens,
                    seq_tokens_orig    = clean_tokens,
                    seq_mask           = batch.get("seq_mask"),
                    cdr_mask           = batch.get("cdr_mask"),
                    coords             = batch.get("coords"),
                    coord_mask         = batch.get("coord_mask"),
                    antigen_coords     = batch.get("antigen_coords"),
                    antigen_coord_mask = batch.get("antigen_coord_mask"),
                    task_mask = {
                        "AR": True,
                        "MLM": False,
                        "GEO": task_mask.get("GEO", False),
                        "IF": task_mask.get("IF", False),
                    },
                    metadata           = batch.get("metadata"),
                    metadata_indices   = batch.get("metadata_indices"),
                    chain_mask         = batch.get("chain_position_mask"),
                    cdr3_lengths       = batch.get("cdr3_lengths"),
                    cdr3_mask          = batch.get("cdr3_mask"),
                    antigen_seq_tokens = batch.get("antigen_seq_tokens"),
                )
                ar_tokens = clean_tokens
            except Exception as e:
                if is_main_process():
                    logger.error(f"Clean AR forward failed (batch {batch_idx}): {e}")
                ar_outputs = outputs
                ar_tokens  = batch.get("seq_tokens")

        # AR metrics
        if task_mask.get("AR", False) and "logits_ar" in outputs:
            valid_rows = (
                batch["sample_task_mask"]["AR"]
                & real_mask
            )

            if valid_rows.sum() > 0:

                logits   = ar_outputs["logits_ar"][valid_rows]
                tokens   = ar_tokens[valid_rows]
                seq_mask = batch["seq_mask"][valid_rows]
            
                targets  = tokens[:, 1:].contiguous()
                seq_mask = seq_mask[:, 1:].contiguous()
            
                min_len  = min(logits.shape[1], targets.shape[1])
                logits   = logits[:, :min_len]
                targets  = targets[:, :min_len]
                seq_mask = seq_mask[:, :min_len]
                
                seq_mask = seq_mask & (targets != 4)
                targets_ar = targets.clone()
                targets_ar[targets == 4] = 0
            
                preds = logits.argmax(dim=-1)

                cdr_mask_raw = batch.get("cdr_mask")
                if cdr_mask_raw is not None:
                    cdr_mask_raw = cdr_mask_raw[valid_rows]
                    cdr_mask_raw = cdr_mask_raw[:, 1:].contiguous()
                    cdr_mask_raw = cdr_mask_raw[:, :targets.shape[1]]
                else:
                    cdr_mask_raw = torch.zeros_like(seq_mask[:, :targets.shape[1]])

                seq_mask_v = seq_mask[:, :targets.shape[1]]
                fw_mask = seq_mask_v.bool() & (~cdr_mask_raw.bool())
                cdr_weighted = seq_mask_v.bool() & cdr_mask_raw.bool()
                
                fw_correct  = ((preds == targets) * fw_mask.bool()).sum()
                fw_total    = fw_mask.sum()
                cdr_correct = ((preds == targets) * cdr_weighted.bool()).sum()
                cdr_total   = cdr_weighted.sum()
            
                metrics_accum["fw_correct"]  += fw_correct.item()
                metrics_accum["fw_total"]    += fw_total.item()
                metrics_accum["cdr_correct"] += cdr_correct.item()
                metrics_accum["cdr_total"]   += cdr_total.item()
            
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets_ar.reshape(-1),
                    ignore_index=0,
                    label_smoothing=0.0,
                    reduction="mean",
                )
                metrics_accum["ar_loss"].append(loss.item())
            
                # Token NLLs
                valid_mask    = seq_mask.bool()
                valid_targets = targets[valid_mask]
                valid_logits  = logits[valid_mask]
                range_mask    = (valid_targets >= 0) & (valid_targets < vocab_size)
                if range_mask.sum() > 0:
                    log_probs       = F.log_softmax(valid_logits[range_mask], dim=-1)
                    token_log_probs = log_probs.gather(1, valid_targets[range_mask].unsqueeze(-1)).squeeze(-1)
                    metrics_accum["ar_token_nlls"].extend((-token_log_probs).cpu().numpy().tolist())
            
                # CDR NLLs                                                       
                if cdr_mask_raw is not None:
                    cdr_pos = seq_mask.bool() & cdr_mask_raw.bool()
                    if cdr_pos.any():
                        cdr_tgt  = targets[cdr_pos]
                        cdr_lgts = logits[cdr_pos]
                        ok = (cdr_tgt >= 0) & (cdr_tgt < logits.size(-1))
                        if ok.sum() > 0:
                            lp = F.log_softmax(cdr_lgts[ok], dim=-1)
                            metrics_accum["cdr_only_nlls"].extend(
                                (-lp.gather(1, cdr_tgt[ok].unsqueeze(-1)).squeeze(-1)).cpu().numpy().tolist()
                            )
            
                # CDR3 accuracy + NLLs                                         
                cdr3_m = batch.get("cdr3_mask")
                if cdr3_m is not None:
                    cdr3_m = cdr3_m[valid_rows]
                    min_len3     = min(logits.shape[1], targets.shape[1])
                    cdr3_shifted = cdr3_m[:, 1:].contiguous()
            
                    l_acc        = min(cdr3_shifted.shape[1], targets.shape[1])
                    cdr3_pos_acc = seq_mask[:, :l_acc].bool() & cdr3_shifted[:, :l_acc].bool()
                    metrics_accum["cdr3_correct"] += ((preds[:, :l_acc] == targets[:, :l_acc]) & cdr3_pos_acc).sum().item()
                    metrics_accum["cdr3_total"]   += cdr3_pos_acc.sum().item()

                    l_nll        = min(cdr3_shifted.shape[1], min_len3)
                    cdr3_pos_nll = seq_mask[:, :l_nll].bool() & cdr3_shifted[:, :l_nll].bool()
                    if cdr3_pos_nll.any():
                        c3_tgt  = targets[:, :l_nll][cdr3_pos_nll]
                        c3_lgts = logits[:, :l_nll][cdr3_pos_nll]
                        ok3 = (c3_tgt >= 0) & (c3_tgt < vocab_size)
                        if ok3.sum() > 0:
                            lp3  = F.log_softmax(c3_lgts[ok3], dim=-1)
                            nlls = (-lp3.gather(1, c3_tgt[ok3].unsqueeze(-1)).squeeze(-1))
                            metrics_accum["cdr3_only_nlls"].extend(nlls.cpu().numpy().tolist())

                            # Split H3 vs L3 using chain_position_mask aligned to AR targets
                            chain_pm = batch.get("chain_position_mask")
                            if chain_pm is not None:
                                chain_pm_v   = chain_pm[valid_rows]
                                chain_shift  = chain_pm_v[:, 1:].contiguous()[:, :l_nll]
                                chain_at_pos = chain_shift[cdr3_pos_nll][ok3]  # 0=heavy, 1=light
                                is_h3 = (chain_at_pos == 0)
                                metrics_accum["h3_only_nlls"].extend(nlls[is_h3].cpu().numpy().tolist())
                                metrics_accum["l3_only_nlls"].extend(nlls[~is_h3].cpu().numpy().tolist())

                    chain_pm = batch.get("chain_position_mask")
                    if chain_pm is not None:
                        chain_pm = chain_pm[valid_rows]
                    chain_shifted_val = (
                        chain_pm[:, 1:].contiguous()[:, :l_acc]
                        if chain_pm is not None else None
                    )

                    for b_idx in range(targets.shape[0]):
                        cdr3_b = cdr3_shifted[b_idx, :l_acc].bool()
                        mask_b = seq_mask[b_idx, :l_acc].bool()
                        valid_b = mask_b & cdr3_b

                        # Per-sample CDR3 accuracy proxy for length adherence.
                        if valid_b.sum() >= 3:
                            per_acc = (
                                (preds[b_idx, :l_acc] == targets[b_idx, :l_acc]) & valid_b
                            ).float().sum() / valid_b.float().sum()
                            metrics_accum["cdr3_per_sample_accs"].append(per_acc.item())

                        # Heavy vs light CDR3 accuracy split.
                        # Growing divergence = DualChainHead successfully specializing.
                        # Flat/zero divergence = heads have collapsed to identical outputs.
                        if chain_shifted_val is not None:
                            chain_b = chain_shifted_val[b_idx]

                            heavy_valid = valid_b & (chain_b == 0)
                            light_valid = valid_b & (chain_b == 1)

                            if heavy_valid.any():
                                metrics_accum["heavy_cdr3_correct"] += (
                                    (preds[b_idx, :l_acc] == targets[b_idx, :l_acc]) & heavy_valid
                                ).sum().item()
                                metrics_accum["heavy_cdr3_total"] += heavy_valid.sum().item()

                            if light_valid.any():
                                metrics_accum["light_cdr3_correct"] += (
                                    (preds[b_idx, :l_acc] == targets[b_idx, :l_acc]) & light_valid
                                ).sum().item()
                                metrics_accum["light_cdr3_total"] += light_valid.sum().item()

        # MLM metrics
        if task_mask.get("MLM", False) and "logits_mlm" in outputs:  # type: ignore
            valid_rows = (
                batch["sample_task_mask"]["MLM"]
                & real_mask
            )

            if valid_rows.sum() > 0:
                # continue

                mlm_logits = outputs["logits_mlm"][valid_rows]

                mlm_labels = batch.get("mlm_labels")
                if mlm_labels is not None:
                    mlm_labels = mlm_labels[valid_rows]

                cdr_mask = batch.get("cdr_mask")
                if cdr_mask is not None:
                    cdr_mask = cdr_mask[valid_rows]
                
                if mlm_labels is not None:
            
                    # align
                    min_len = min(mlm_logits.shape[1], mlm_labels.shape[1])
                    mlm_logits = mlm_logits[:, :min_len]
                    mlm_labels = mlm_labels[:, :min_len]
            
                    if cdr_mask is not None:
                        cdr_mask = cdr_mask[:, :min_len]
            
                    mlm_mask = mlm_labels != -100
                    
                    if mlm_mask.sum() > 0:
                        # Accuracy
                        preds = mlm_logits.argmax(dim=-1)
                        metrics_accum["mlm_correct"] += (preds[mlm_mask] == mlm_labels[mlm_mask]).sum().item()
                        metrics_accum["mlm_total"] += mlm_mask.sum().item()
                        
                        # Loss
                        loss = F.cross_entropy(
                            mlm_logits.reshape(-1, mlm_logits.size(-1)),
                            mlm_labels.reshape(-1),
                            label_smoothing=0.0,
                            ignore_index=-100,
                            reduction="mean",
                        )
                        metrics_accum["mlm_loss"].append(loss.item())
                        
                        # Token NLL
                        valid_labels = mlm_labels[mlm_mask]
                        valid_logits = mlm_logits[mlm_mask]
                        
                        mlm_vocab_size = mlm_logits.size(-1)
                        valid_mask = (valid_labels >= 0) & (valid_labels < mlm_vocab_size)
                        
                        if valid_mask.sum() > 0:
                            filtered_labels = valid_labels[valid_mask]
                            filtered_logits = valid_logits[valid_mask]
                            
                            log_probs = F.log_softmax(filtered_logits, dim=-1)
                            token_log_probs = log_probs.gather(1, filtered_labels.unsqueeze(-1)).squeeze(-1)
                            token_nlls = -token_log_probs
            
                            metrics_accum["mlm_token_nlls"].extend(token_nlls.cpu().numpy().tolist())

        # GEO metrics
        geo_source = ar_outputs if "pred_coords" in ar_outputs else outputs
        if task_mask.get("GEO", False) and "pred_coords" in geo_source:
            valid_rows = (
                batch["sample_task_mask"]["GEO"]
                & real_mask
            )

            if valid_rows.sum() > 0:

                pred = geo_source["pred_coords"][valid_rows]

                target = batch.get("coords")
                if target is not None:
                    target = target[valid_rows]

                mask = batch.get("coord_mask")
                if mask is not None:
                    mask = mask[valid_rows]

                if pred is not None and target is not None:
                    # Handle multi-atom case
                    if target.ndim == 4:
                        if target.shape[2] >= 2:
                            target = target[:, :, 1, :]  # Use Cα (index 1)
                        else:
                            target = target.squeeze(2)
                        
                        if mask is not None and mask.ndim == 3:
                            if mask.shape[2] >= 2:
                                mask = mask[:, :, 1]
                            else:
                                mask = mask.squeeze(2)

                    # Handle length mismatch
                    L = min(pred.shape[1], target.shape[1])
                    pred, target = pred[:, :L], target[:, :L]
                    if mask is not None:
                        mask = mask[:, :L]
                    if mask is not None and mask.ndim == 3:
                        mask = mask.any(dim=2)
    
                    # Centre coordinates per sample (removes absolute-position offset)
                    if mask is not None:
                        m     = mask.float().unsqueeze(-1)
                        denom = m.sum(dim=1, keepdim=True).clamp(min=1)
                        target = target - (target * m).sum(dim=1, keepdim=True) / denom
                        pred   = pred   - (pred   * m).sum(dim=1, keepdim=True) / denom
                    else:
                        target = target - target.mean(dim=1, keepdim=True)
                        pred   = pred   - pred.mean(dim=1, keepdim=True)
    
                    rmsd_aligned   = compute_aligned_rmsd(pred, target, mask)
                    rmsd_unaligned = compute_unaligned_rmsd(pred, target, mask)
                    ca_ca_stats    = compute_ca_ca_distances(pred, mask)
    
                    if torch.isfinite(rmsd_aligned):
                        metrics_accum["geo_rmsd_aligned"].append(rmsd_aligned.item())
                    if torch.isfinite(rmsd_unaligned):
                        metrics_accum["geo_rmsd_unaligned"].append(rmsd_unaligned.item())
                    if torch.isfinite(ca_ca_stats['mean']):
                        metrics_accum["geo_ca_ca_mean"].append(ca_ca_stats['mean'].item())
                    if torch.isfinite(ca_ca_stats['std']):
                        metrics_accum["geo_ca_ca_std"].append(ca_ca_stats['std'].item())
                    if torch.isfinite(ca_ca_stats['deviation']):
                        metrics_accum["geo_ca_ca_deviation"].append(ca_ca_stats['deviation'].item())


        # Interface metrics
        if task_mask.get("IF", False):
            # Sequence-level binding metrics 
            if_source = ar_outputs if (
                ("binding_logits" in ar_outputs) or
                ("residue_interface_logits" in ar_outputs)
            ) else outputs

            if "binding_logits" in if_source:
                valid_rows = (
                    batch["sample_task_mask"]["IF"]
                    & real_mask
                )

                if valid_rows.sum() > 0:

                    iface_logits_raw = if_source["binding_logits"][valid_rows]

                    iface_labels = batch.get("interface_labels")
                    if iface_labels is not None:
                        iface_labels = iface_labels[valid_rows]
                    
                    if iface_labels is not None:
                        if iface_logits_raw.dim() > 1:
                            iface_logits_raw = iface_logits_raw.squeeze(-1)
                        if iface_labels.dim() > 1:
                            iface_labels = iface_labels.squeeze(-1)
                        
                        iface_logits = torch.sigmoid(iface_logits_raw)
                        iface_labels = iface_labels

                        valid = torch.isfinite(iface_logits) & torch.isfinite(iface_labels)

                        iface_logits = iface_logits[valid].cpu()
                        iface_labels = iface_labels[valid].cpu()

                        if iface_logits.numel() > 0:
                            if iface_logits.dim() == 0:
                                metrics_accum["iface_logits"].append(iface_logits.item())
                                metrics_accum["iface_labels"].append(iface_labels.item())
                            else:
                                metrics_accum["iface_logits"].extend(iface_logits.tolist())
                                metrics_accum["iface_labels"].extend(iface_labels.tolist())

            # Residue-level
            if "residue_interface_logits" in if_source:
                valid_rows = (
                    batch["sample_task_mask"]["IF"]
                    & real_mask
                )

                if valid_rows.sum() > 0:

                    res_logits = if_source["residue_interface_logits"][valid_rows]

                    res_masks = batch.get("interface_mask")
                    if res_masks is not None:
                        res_masks = res_masks[valid_rows]
                    
                    if res_masks is not None:
                        for i in range(res_logits.shape[0]):
                            mask = res_masks[i]
                            if mask is not None and isinstance(mask, torch.Tensor):
                                seq_mask_i = batch["seq_mask"][valid_rows][i].bool()
                                mask_i = mask[:len(seq_mask_i)]

                                valid = seq_mask_i[:len(mask_i)] & torch.isfinite(mask_i)

                                if valid.any():
                                    sample_logits = torch.sigmoid(res_logits[i, :len(mask_i), 0])

                                    metrics_accum["residue_iface_logits"].extend(
                                        sample_logits[valid].cpu().tolist()
                                    )
                                    metrics_accum["residue_iface_labels"].extend(
                                        mask_i[valid].cpu().tolist()
                                    )

    # Aggregate metrics
    results = {}
    
    # AR metrics
    fw_total    = metrics_accum["fw_total"]
    cdr_total   = metrics_accum["cdr_total"]
    fw_correct  = metrics_accum["fw_correct"]
    cdr_correct = metrics_accum["cdr_correct"]
    
    if fw_total > 0:
        results["fw_acc"] = fw_correct / fw_total
    
    if cdr_total > 0:
        results["cdr_acc"] = cdr_correct / cdr_total
    
    total_tokens = fw_total + cdr_total
    total_correct = fw_correct + cdr_correct
    
    if total_tokens > 0:
        results["ar_acc"] = total_correct / total_tokens
    
    if metrics_accum["ar_loss"]:
        avg_ar_loss = sum(metrics_accum["ar_loss"]) / len(metrics_accum["ar_loss"])
        results["ar_ppl"] = np.exp(avg_ar_loss)  # Perplexity
        results["ar_loss"] = avg_ar_loss / max_ce  # NORMALIZED
    
    # AR NLL (mean negative log-likelihood per token)
    if metrics_accum["ar_token_nlls"]:
        results["ar_nll"] = np.mean(metrics_accum["ar_token_nlls"])

    if metrics_accum["cdr3_total"] > 0:
        results["cdr3_acc"] = metrics_accum["cdr3_correct"] / metrics_accum["cdr3_total"]
    
    # CDR3 length adherence proxy.
    # Fraction of samples where per-sample CDR3 accuracy > 0.70.
    # Interpretation: at >0.60 the model reliably generates correct CDR3 length.
    # Target from competitive checklist: >0.60.
    if metrics_accum["cdr3_per_sample_accs"]:
        per_sample = metrics_accum["cdr3_per_sample_accs"]
        results["cdr3_mean_per_sample_acc"]      = float(np.mean(per_sample))
        results["cdr3_length_adherence_proxy"]   = float(np.mean([a > 0.70 for a in per_sample]))

    # Heavy vs light chain CDR3 accuracy and divergence.
    # chain_cdr3_divergence should grow as DualChainHead specializes.
    # A value near 0 throughout training indicates head collapse.
    if metrics_accum["heavy_cdr3_total"] > 0:
        results["heavy_cdr3_acc"] = (
            metrics_accum["heavy_cdr3_correct"] / metrics_accum["heavy_cdr3_total"]
        )
    if metrics_accum["light_cdr3_total"] > 0:
        results["light_cdr3_acc"] = (
            metrics_accum["light_cdr3_correct"] / metrics_accum["light_cdr3_total"]
        )
    if "heavy_cdr3_acc" in results and "light_cdr3_acc" in results:
        results["chain_cdr3_divergence"] = abs(
            results["heavy_cdr3_acc"] - results["light_cdr3_acc"]
        )
    
    if metrics_accum["cdr_only_nlls"]:
        results["cdr_ppl"] = float(np.exp(np.mean(metrics_accum["cdr_only_nlls"])))
 
    if metrics_accum["cdr3_only_nlls"]:
        results["cdr3_ppl"] = float(np.exp(np.mean(metrics_accum["cdr3_only_nlls"])))
        
    if metrics_accum["h3_only_nlls"]:
        results["h3_ppl"] = float(np.exp(np.mean(metrics_accum["h3_only_nlls"])))
    if metrics_accum["l3_only_nlls"]:
        results["l3_ppl"] = float(np.exp(np.mean(metrics_accum["l3_only_nlls"])))
    
    # MLM metrics  
    if metrics_accum["mlm_total"] > 0:
        results["mlm_acc"] = metrics_accum["mlm_correct"] / metrics_accum["mlm_total"]
    
    if metrics_accum["mlm_loss"]:
        avg_mlm_loss = sum(metrics_accum["mlm_loss"]) / len(metrics_accum["mlm_loss"])
        results["mlm_loss"] = avg_mlm_loss / max_ce  # NORMALIZED
    
    # MLM NLL
    if metrics_accum["mlm_token_nlls"]:
        results["mlm_nll"] = np.mean(metrics_accum["mlm_token_nlls"])
    
    # GEO metrics
    if metrics_accum["geo_rmsd_aligned"]:
        # Primary metric: Aligned RMSD
        avg_rmsd_aligned = np.mean(metrics_accum["geo_rmsd_aligned"])
        results["geo_rmsd"] = avg_rmsd_aligned  # Main metric (aligned!)
        
        # Normalized loss (for training)
        results["geo_loss"] = 1.0 - np.exp(-avg_rmsd_aligned / 15.0)
        
        # Additional metrics
        avg_rmsd_unaligned = np.mean(metrics_accum["geo_rmsd_unaligned"])
        results["geo_rmsd_unaligned"] = avg_rmsd_unaligned
        
        # (TODO) remove: Alignment improvement (shows how much Kabsch helps)
        if avg_rmsd_unaligned > 0:
            improvement = (avg_rmsd_unaligned - avg_rmsd_aligned) / avg_rmsd_unaligned
            results["geo_alignment_benefit"] = improvement
        
        # Cα-Cα distance quality
        if metrics_accum["geo_ca_ca_mean"]:
            results["geo_ca_ca_mean"] = np.mean(metrics_accum["geo_ca_ca_mean"])
            results["geo_ca_ca_std"] = np.mean(metrics_accum["geo_ca_ca_std"])
            results["geo_ca_ca_deviation"] = np.mean(metrics_accum["geo_ca_ca_deviation"])

    
    # Interface metrics
    # Sequence-level binding metrics (SAbDab)
    seq_labels = metrics_accum["iface_labels"]
    seq_logits = metrics_accum["iface_logits"]
    
    if seq_labels and len(seq_labels) > 1:
        num_pos = sum(seq_labels)
        num_neg = len(seq_labels) - num_pos
        
        if num_pos > 0 and num_neg > 0:
            try:
                results["iface_roc_auc"] = roc_auc_score(seq_labels, seq_logits)
                results["iface_pr_auc"] = average_precision_score(seq_labels, seq_logits)
            except Exception as e:
                if is_main_process():
                    logger.error(f"Error computing sequence-level interface metrics: {e}")
    
    # Residue-level interface metrics (SAbDab with structure)
    res_labels = metrics_accum["residue_iface_labels"]
    res_logits = metrics_accum["residue_iface_logits"]
    
    if res_labels and len(res_labels) > 1:
        num_pos = sum(res_labels)
        num_neg = len(res_labels) - num_pos
        
        if num_pos > 0 and num_neg > 0:
            try:
                results["residue_iface_roc_auc"] = roc_auc_score(res_labels, res_logits)
                results["residue_iface_pr_auc"] = average_precision_score(res_labels, res_logits)
                
                # Additional residue-level metrics
                # Precision/Recall at different thresholds
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(res_labels, res_logits)
                
                # F1 score at optimal threshold
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1_idx = np.argmax(f1_scores)
                results["residue_iface_f1"] = f1_scores[best_f1_idx]
                results["residue_iface_precision"] = precision[best_f1_idx]
                results["residue_iface_recall"] = recall[best_f1_idx]
                
            except Exception as e:
                if is_main_process():
                    logger.error(f"Error computing residue-level interface metrics: {e}")

    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return results