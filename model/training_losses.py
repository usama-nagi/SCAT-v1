# model/training_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math
from utils.geometry import compute_aligned_rmsd
import torch.distributed as dist


def ddp_align_len(*tensors):
    min_len = min(t.shape[1] for t in tensors if t is not None)
    return [t[:, :min_len] if t is not None else None for t in tensors]

def antigen_infonce_loss(antigen_pooled: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """AgForce-style pairwise InfoNCE within a batch."""
    B = antigen_pooled.shape[0]
    if B < 2:
        return antigen_pooled.sum() * 0.0
    z = F.normalize(antigen_pooled.float(), dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    labels = torch.arange(B, device=antigen_pooled.device)
    return F.cross_entropy(sim, labels)

class SCATLoss(nn.Module):
    """
    Unified multi-task loss for SCAT with NORMALIZED 0-1 scale.
    All sequence losses (AR, MLM) forced to FP32 to prevent FP16 overflow
    on hard CDR positions during AMP training.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        loss_weights: Dict[str, float],
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.w_ar  = loss_weights.get("AR",  1.0)
        self.w_mlm = loss_weights.get("MLM", 1.0)
        self.w_geo = loss_weights.get("GEO", 1.0)
        self.w_if  = loss_weights.get("IF",  1.0)
        self.weights = dict(loss_weights)
        self.cdr_w       = loss_weights.get("CDR_W",        1.5)
        self.cdr3_extra  = loss_weights.get("CDR3_EXTRA",   1.5)
        self.cdr3_extra_h  = loss_weights.get("CDR3_EXTRA_H",   1.5)
        self.cdr3_extra_l  = loss_weights.get("CDR3_EXTRA_L",   1.5)
        self.cdr3_contact_extra = loss_weights.get("CDR3_CONTACT_EXTRA", 0.0)

        # Python float - preserves gradients
        self.max_ce = math.log(vocab_size)
        print(f"[SCATLoss] Vocab size: {vocab_size}, Max CE: {self.max_ce:.4f}")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with individual losses + total (all normalized to ~0-1)."""

        losses = {}
        device = self._device(outputs)
        total_loss = None

        task_mask = batch["task_mask"]

        # Autoregressive loss
        if "logits_ar" in outputs:
            loss_ar = self._ar_loss(outputs, batch)
            if loss_ar is not None:
                losses["loss_ar"] = loss_ar
                w_ar = self.weights.get("AR", self.w_ar)
                total_loss = w_ar * loss_ar if total_loss is None else total_loss + w_ar * loss_ar

        # MLM loss
        if "logits_mlm" in outputs:
            loss_mlm = self._mlm_loss(outputs, batch)
            if loss_mlm is not None:
                losses["loss_mlm"] = loss_mlm
                w_mlm = self.weights.get("MLM", self.w_mlm)
                total_loss = w_mlm * loss_mlm if total_loss is None else total_loss + w_mlm * loss_mlm

        # Geometry loss
        if "pred_coords" in outputs:
            loss_geo = self._geo_loss(outputs, batch)
            if loss_geo is not None:
                losses["loss_geo"] = loss_geo
                w_geo = self.weights.get("GEO", self.w_geo)
                total_loss = w_geo * loss_geo if total_loss is None else total_loss + w_geo * loss_geo

        # Interface loss (binding + residue-level interface)
        if ("binding_logits" in outputs or
                "residue_interface_logits" in outputs):
            loss_if = self._interface_loss(outputs, batch)
            if loss_if is not None:
                losses["loss_if"] = loss_if
                w_if = self.weights.get("IF", self.w_if)
                total_loss = w_if * loss_if if total_loss is None else total_loss + w_if * loss_if
        
        # DDP unused-param zero-cost pass-through
        _z = outputs.get("_unused_touch")
        if _z is not None:
            total_loss = _z if total_loss is None else total_loss + _z

        losses["total"] = total_loss if total_loss is not None else torch.tensor(0.0, device=device)
        losses["_invalid"] = (total_loss is None)
        return losses

    def _cdr3_contact_weight(self, c3, contact, shift):
        """
        c3:      [B, T] float CDR3 mask, already length-aligned to the loss tensor.
        contact: [B, L] antigen-contact mask (1.0 at contact positions), or None.
        shift:   True for AR (drop first token to match teacher-forced targets).
        Returns additive weight [B, T]: contact-CDR3 positions get
        + self.cdr3_contact_extra, everything else gets +0.
        No-ops (returns zeros) if contact is None or weight is 0 - safe default,
        existing behavior unchanged unless CDR3_CONTACT_EXTRA is set in config.
        """
        if contact is None or self.cdr3_contact_extra == 0.0:
            return torch.zeros_like(c3)
        co = contact[:, 1:] if shift else contact
        T  = c3.shape[1]
        co = co[:, :T].float()
        if co.shape[1] < T:
            co = torch.cat(
                [co, torch.zeros(co.shape[0], T - co.shape[1], device=c3.device)],
                dim=1,
            )
        return c3 * co * self.cdr3_contact_extra
    
    def _cdr3_chain_weight(self, c3, chain, shift):
        """
        c3:    [B, T] float CDR3 mask, already length-aligned to the loss tensor.
        chain: [B, L] chain position mask (0=heavy, 1=light), or None.
        shift: True for AR (drop first token to match teacher-forced targets).
        Returns additive weight [B, T]: H3 * cdr3_extra_h + L3 * cdr3_extra_l.
        Falls back to uniform cdr3_extra when chain is None.
        """
        if chain is None:
            return c3 * self.cdr3_extra
        ch = chain[:, 1:] if shift else chain
        T  = c3.shape[1]
        ch = ch[:, :T].float()
        if ch.shape[1] < T:
            ch = torch.cat(
                [ch, torch.zeros(ch.shape[0], T - ch.shape[1], device=c3.device)],
                dim=1,
            )
        is_heavy = (ch < 0.5).float()
        is_light = 1.0 - is_heavy
        return c3 * (is_heavy * self.cdr3_extra_h + is_light * self.cdr3_extra_l)
    
    def _ar_loss(self, outputs, batch):
        """
        Normalized AR cross entropy.
        """
        if not batch["task_mask"].get("AR", False):
            return outputs["logits_ar"].sum() * 0.0
        
        with torch.amp.autocast('cuda', enabled=False):  # type: ignore
            logits   = outputs["logits_ar"].float()       # explicit FP32 cast
            # AR targets must be the clean (pre-MLM) sequence. seq_tokens is
            # MLM-corrupted on OAS, which both mislabels ~15% of positions as
            # MASK and drops them from supervision via the (targets != 4) filter
            # below - and since OAS MLM masking is CDR-biased, that filter
            # disproportionately removes CDR3 positions.
            tokens   = batch.get("seq_tokens_orig")
            if tokens is None:
                tokens = batch["seq_tokens"]
            mask     = batch["seq_mask"]
            cdr_mask = batch.get("cdr_mask")

            targets = tokens[:, 1:].contiguous()
            mask    = mask[:, 1:].contiguous()
            
            valid_rows = batch["sample_task_mask"]["AR"] & ~batch["synthetic_negative_mask"]

            logits  = logits[valid_rows]
            targets = targets[valid_rows]
            mask    = mask[valid_rows]

            if cdr_mask is not None:
                cdr_mask = cdr_mask[valid_rows]
                cdr_mask = cdr_mask[:, 1:].contiguous()

            logits, targets, mask, cdr_mask = ddp_align_len(
                            logits, targets, mask, cdr_mask)
            
            mask = mask * (targets != 4)

            ar_smoothing = self.weights.get("AR_LABEL_SMOOTH", 0.05)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_idx,
                reduction="none",
                label_smoothing=ar_smoothing,
            )
            loss = loss.view(targets.shape)

            token_w = torch.ones_like(mask, dtype=torch.float)

            if cdr_mask is not None:
                full_cdr = cdr_mask.float()     # CDR1+2+3 combined
                token_w  = token_w + full_cdr * self.cdr_w   # CDR1+2+3 -> 2.5×

                # CDR3-specific upweight
                cdr3_raw = batch.get("cdr3_mask")
                if cdr3_raw is not None:
                    cdr3_raw = cdr3_raw[valid_rows]
                    c3 = cdr3_raw[:, 1:].contiguous()              # shift for AR
                    min_l = min(c3.shape[1], loss.shape[1])
                    c3    = c3[:, :min_l].float()
                    if min_l < loss.shape[1]:
                        c3 = torch.cat(
                            [c3, torch.zeros(c3.shape[0], loss.shape[1] - min_l,
                                             device=c3.device)], dim=1
                        )
                    chain_pm = batch.get("chain_position_mask")
                    if chain_pm is not None:
                        chain_pm = chain_pm[valid_rows]
                    token_w = token_w + self._cdr3_chain_weight(
                        c3, chain_pm, shift=True
                    )

                    contact_raw = batch.get("contact_mask")
                    if contact_raw is not None:
                        contact_raw = contact_raw[valid_rows]
                    token_w = token_w + self._cdr3_contact_weight(
                        c3, contact_raw, shift=True
                    )

            # Downweight framework on SAbDab - prevents germline memorization
            # from dominating CDR gradient budget
            fw_scale = self.weights.get("AR_FW_WEIGHT_SABDAB", 1.0)
            if fw_scale < 1.0 and cdr_mask is not None and batch.get("source") == "sabdab":
                fw_mask = (~cdr_mask.bool()).float()   # 1 where framework, 0 where CDR
                # align to token_w length (already shifted by 1 for AR)
                min_l = min(fw_mask.shape[1], token_w.shape[1])
                fw_mask = fw_mask[:, :min_l]
                if min_l < token_w.shape[1]:
                    fw_mask = torch.cat([fw_mask,
                        torch.zeros(fw_mask.shape[0], token_w.shape[1] - min_l,
                                    device=fw_mask.device)], dim=1)
                token_w = token_w * (1.0 - fw_mask * (1.0 - fw_scale))
            
            token_w = token_w * mask.float()   # zero out padding

            n_tokens = token_w.sum()
            if n_tokens == 0:
                return (logits * 0.0).sum()

            raw_loss = (loss * token_w).sum() / n_tokens.clamp(min=1.0)
            normalized_loss = raw_loss / self.max_ce
            return normalized_loss

    def _mlm_loss(self, outputs, batch):
        """
        Normalized MLM cross entropy.
        """
        if ("logits_mlm" not in outputs or outputs["logits_mlm"] is None
                or not batch["task_mask"].get("MLM", False)
                or batch.get("mlm_labels") is None):
            logits = outputs.get("logits_mlm")
            if logits is not None:
                return logits.sum() * 0.0
            device = next(iter(outputs.values())).device if outputs else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True)

        with torch.amp.autocast('cuda', enabled=False):  # type: ignore
            logits   = outputs["logits_mlm"].float()      # explicit FP32 cast
            labels   = batch.get("mlm_labels")
            cdr_mask = batch.get("cdr_mask")

            if labels is None:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            valid_rows = batch["sample_task_mask"]["MLM"] & ~batch["synthetic_negative_mask"]

            logits = logits[valid_rows]
            labels = labels[valid_rows]

            if cdr_mask is not None:
                cdr_mask = cdr_mask[valid_rows]

            logits, labels, cdr_mask = ddp_align_len(
                            logits, labels, cdr_mask
                        )

            mlm_mask = labels != -100

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=0.1,
            )
            loss = loss.view(labels.shape)
            
            if cdr_mask is not None:
                assert cdr_mask.shape == logits.shape[:2], f"Shape mismatch after align: {cdr_mask.shape} vs {logits.shape[:2]}"

            if cdr_mask is not None:
                # CDR1+2+3: 2.5×, CDR3: 5.0×, framework: 1.0×
                # Consistent with _ar_loss weighting.
                # MLM labels are at the same positions as the input tokens.
                base_w = 1.0 + cdr_mask.float() * self.cdr_w

                cdr3_raw = batch.get("cdr3_mask")
                if cdr3_raw is not None:
                    cdr3_raw = cdr3_raw[valid_rows]
                    min_l = min(cdr3_raw.shape[1], loss.shape[1])
                    c3    = cdr3_raw[:, :min_l].float()
                    if min_l < loss.shape[1]:
                        c3 = torch.cat(
                            [c3, torch.zeros(c3.shape[0], loss.shape[1] - min_l,
                                             device=c3.device)], dim=1
                        )
                    chain_pm = batch.get("chain_position_mask")
                    if chain_pm is not None:
                        chain_pm = chain_pm[valid_rows]
                    base_w = base_w + self._cdr3_chain_weight(
                        c3, chain_pm, shift=False
                    )

                    contact_raw = batch.get("contact_mask")
                    if contact_raw is not None:
                        contact_raw = contact_raw[valid_rows]
                    base_w = base_w + self._cdr3_contact_weight(
                        c3, contact_raw, shift=False
                    )

                weights = mlm_mask.float() * base_w
            else:
                weights = mlm_mask.float()

            loss = loss * weights

            loss_sum   = loss.sum()
            num_tokens = weights.sum()
            
            if num_tokens == 0:
                return (logits * 0.0).sum()

            num_tokens = num_tokens.float()
            raw_loss = loss_sum / num_tokens.clamp(min=1.0)
            normalized_loss = raw_loss / self.max_ce
            return normalized_loss

    def _geo_loss(self, outputs, batch):
        """
        Normalized geometry loss using Kabsch-aligned RMSD.
        FP32 forced - SVD numerical stability.
        """
        pred = outputs.get("pred_coords")
        true = batch.get("coords")
        mask = batch.get("coord_mask")
        
        if not batch["task_mask"].get("GEO", False):
            return outputs["pred_coords"].sum() * 0.0

        if pred is None or true is None:
            return torch.tensor(0.0, device=self._device(outputs), requires_grad=True)
        
        valid_rows = batch["sample_task_mask"]["GEO"]

        pred = pred[valid_rows]
        true = true[valid_rows]

        if mask is not None:
            mask = mask[valid_rows]
            
        if pred.shape[0] == 0:
            return outputs["pred_coords"].sum() * 0.0

        with torch.amp.autocast('cuda', enabled=False):  # type: ignore
            pred = pred.float()
            true = true.float()

            # Cα extraction
            if true.ndim == 4:
                true = true[:, :, 1, :] if true.shape[2] >= 2 else true.squeeze(2)
                if mask is not None and mask.ndim == 3:
                    mask = mask[:, :, 1] if mask.shape[2] >= 2 else mask.squeeze(2)

            L = min(pred.shape[1], true.shape[1])
            pred, true = pred[:, :L], true[:, :L]
            if mask is not None:
                mask = mask[:, :L]
            if mask is not None and mask.ndim == 3:
                mask = mask.any(dim=2)

            # Centre per sample - removes absolute coordinate offset
            if mask is not None:
                m     = mask.float().unsqueeze(-1)
                denom = m.sum(dim=1, keepdim=True).clamp(min=1)
                true  = true - (true * m).sum(dim=1, keepdim=True) / denom
                pred  = pred - (pred * m).sum(dim=1, keepdim=True) / denom
            else:
                true = true - true.mean(dim=1, keepdim=True)
                pred = pred - pred.mean(dim=1, keepdim=True)

            aligned_rmsd = compute_aligned_rmsd(pred, true, mask)
            if not torch.isfinite(aligned_rmsd):
                return torch.zeros((), device=pred.device, requires_grad=True)

            normalized_loss = 1.0 - torch.exp(-aligned_rmsd / 15.0)

        return normalized_loss

    def _interface_loss(self, outputs, batch):
        """Handles both sequence-level binding and residue-level interface."""
        
        if not batch["task_mask"].get("IF", False):
            z = torch.zeros([], device=self._device(outputs))
            if "binding_logits" in outputs:
                z = z + outputs["binding_logits"].sum() * 0.0
            if "residue_interface_logits" in outputs:
                z = z + outputs["residue_interface_logits"].sum() * 0.0
            return z
        
        if_losses = []

        # Sequence-level binding
        if "binding_logits" in outputs:
            logits = outputs["binding_logits"]
            labels = batch.get("interface_labels")
            
            valid_rows = batch["sample_task_mask"]["IF"]

            logits = logits[valid_rows]

            if labels is not None:
                labels = labels[valid_rows]

            if labels is not None:
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)

                logits = logits.float()
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = logits.clamp(-20.0, 20.0)
                
                labels = labels.float().clamp(0.0, 1.0)

                valid_mask = torch.isfinite(labels) & torch.isfinite(logits)
                labels_used = labels

                # conditionally compute loss
                if valid_mask.sum() > 0:
                    logits_v = logits[valid_mask]
                    labels_v = labels_used[valid_mask]

                    if logits_v.numel() > 0:
                        if_losses.append(
                            F.binary_cross_entropy_with_logits(logits_v, labels_v)
                        )

        # Residue-level interface
        if "residue_interface_logits" in outputs:
            logits = outputs["residue_interface_logits"]  # [B, L, 1]
            masks  = batch.get("interface_mask")
            
            valid_rows = batch["sample_task_mask"]["IF"]

            logits = logits[valid_rows]

            if masks is not None:
                masks = masks[valid_rows]

            if isinstance(masks, torch.Tensor):
                logits = logits[..., 0]  # [B, L]
                seq_mask = batch["seq_mask"][valid_rows]

                L = min(logits.shape[1], masks.shape[1])
                logits = logits[:, :L]
                masks  = masks[:, :L]

                valid = seq_mask.bool() & torch.isfinite(masks)
                valid_count = valid.sum()

                if valid_count.item() > 5:
                    logits_v = logits[valid].float().clamp(-30.0, 30.0)
                    labels_v = masks[valid].float().clamp(0.0, 1.0)
                    if_losses.append(F.binary_cross_entropy_with_logits(logits_v, labels_v))
                else:
                    if_losses.append(logits.sum() * 0.0)
            else:
                if_losses.append(logits[..., 0].sum() * 0.0)


        if len(if_losses) > 0:
            if_loss = torch.stack(if_losses).mean()
        else:
            if_loss = torch.zeros((), device=self._device(outputs), requires_grad=True)

        MAX_BCE = math.log(2)
        return if_loss / MAX_BCE
    
    @staticmethod
    def _device(outputs: Dict[str, torch.Tensor]) -> torch.device:
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")