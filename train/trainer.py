# train/trainer.py
import os
import sys
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as _F
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from utils.logging import get_logger
from train.validator import validate
from collections import defaultdict
from datetime import datetime


logger = get_logger(__name__)

DEBUG_OFFSETS = [
    1, 2, 10, 50, 100, 200,
    1000, 1500, 2000, 2500,
    3000, 3500, 4000, 4500
]

def _all_ranks_agree_skip(should_skip: bool, device) -> bool:
    """
    In DDP every rank MUST execute the same number of forward+backward calls.
    If ANY rank wants to skip a step (bad batch, None batch, invalid loss),
    ALL ranks must skip together - otherwise the ranks that don't skip will
    call backward() and hang waiting for the gradient all_reduce that the
    skipping rank never initiates.
 
    Returns True if any rank voted to skip this step.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return should_skip
 
    flag = torch.tensor(1.0 if should_skip else 0.0, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0.0

class SCATTrainer:
    """
    Multi-task trainer for SCAT model with probability-based dataset sampling.
    
    APPROACH:
    - Sample ONE dataset per step (based on probabilities)
    - Single forward/backward per step
    - cleaner optimization geometry
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        mixed_sampler,
        val_loaders,
        device,
        cfg,
        scheduler=None,
        rank=0,              
        world_size=1,        
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.mixed_sampler = mixed_sampler
        self.val_loaders = val_loaders
        self.device = device
        self.cfg = cfg
        self.rank = rank                    
        self.world_size = world_size        
        self.is_main = (rank == 0)          
        
        self.grad_clip = cfg["train"].get("grad_clip", 1.0)
        self.log_interval = cfg["train"].get("log_interval", 50)

        self.use_amp = cfg["train"].get("use_amp", False)
        self.scaler = GradScaler('cuda', init_scale=1024, growth_factor=1.5,
                          backoff_factor=0.5, growth_interval=4000) if self.use_amp else None
        
        # Get gradient weights from config (should be near 1.0)
        self.grad_weights = cfg["train"].get("gradient_weights", {})
        self._consecutive_failures = defaultdict(int)
        
        if self.grad_weights and self.is_main:
            logger.info("[Trainer] Gradient weights:")
            for ds, weight in self.grad_weights.items():
                logger.info(f"  {ds}: {weight}")
                if weight > 2.0:
                    logger.warning(f"  WARNING: Large gradient weight for {ds}: {weight}")
                    logger.warning("  Consider using sampling probability instead")

        if self.is_main:
            logger.info(f"[Trainer] Initialized (rank {rank}/{world_size})")
            logger.info(f"[Trainer] AMP: {'enabled' if self.use_amp else 'disabled'}")
            logger.info(f"[Trainer] Gradient clipping: {self.grad_clip}")

    def train_epoch(
        self, 
        epoch, 
        global_step=0,                      
        max_steps=float('inf'),             
        val_check_interval=None,            
        save_every_n_steps=None,            
        ckpt_dir=None,                      
        best_val_loss=float('inf')
    ):
        """
        Train one epoch with probability-based dataset sampling.
        
        1. Sample one dataset (based on probabilities)
        2. Forward pass
        3. Backward pass
        4. Optimizer step
        """
        self.model.train()
        
        import time
        last_successful_step_time = time.time()
        TIMEOUT_SECONDS = 1200 
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # steps for this epoch
        steps_per_epoch = self.cfg["train"].get("steps_per_epoch", 10000)
        
        pbar = tqdm(
            range(steps_per_epoch), 
            desc=f"Epoch {epoch}", 
            disable=(not sys.stdout.isatty() or not self.is_main)
        )
    
        # Stats per dataset
        dataset_stats = defaultdict(lambda: {
            "total": 0.0,
            "ar": 0.0,
            "mlm": 0.0,
            "geo": 0.0,
            "if": 0.0,
            "anti_inv": 0.0,
            "anti_inv_kl": 0.0,
            "anti_inv_nll_real": 0.0,
            "anti_inv_nll_shuf": 0.0,
            "anti_inv_nll_rank": 0.0,
            "anti_inv_infonce": 0.0,
            "anti_inv_count": 0,
            "fw_correct": 0,
            "fw_total": 0,
            "cdr_correct": 0,
            "cdr_total": 0,
            "h3_correct": 0,
            "h3_total":   0,
            "l3_correct": 0,
            "l3_total":   0,
            "h3_contact_correct": 0,
            "h3_contact_total":   0,
            "h3_noncontact_correct": 0,
            "h3_noncontact_total":   0,
            "cdr3_correct": 0,
            "cdr3_total":   0,
            "mlm_correct": 0,
            "mlm_total": 0,
            "count": 0,
            "skipped": 0,
            "errors": 0,
        })
    
        overall_stats = {
            "total": 0.0,
            "ar": 0.0,
            "mlm": 0.0,
            "geo": 0.0,
            "if": 0.0,
            "anti_inv": 0.0,
            "anti_inv_kl": 0.0,
            "anti_inv_nll_real": 0.0,
            "anti_inv_nll_shuf": 0.0,
            "anti_inv_nll_rank": 0.0,
            "anti_inv_infonce": 0.0,
            "anti_inv_count": 0,
            "fw_correct": 0,
            "fw_total": 0,
            "cdr_correct": 0,
            "cdr_total": 0,
            "h3_correct": 0,
            "h3_total":   0,
            "l3_correct": 0,
            "l3_total":   0,
            "h3_contact_correct": 0,
            "h3_contact_total":   0,
            "h3_noncontact_correct": 0,
            "h3_noncontact_total":   0,
            "cdr3_correct": 0,
            "cdr3_total":   0,
            "mlm_correct": 0,
            "mlm_total": 0,
        }
        
        resume_step = global_step  # current checkpoint step
    
        for step_in_epoch in pbar:
            global_step += 1
            
            if time.time() - last_successful_step_time > TIMEOUT_SECONDS:
                if self.is_main:
                    logger.error(f"Training stuck for {TIMEOUT_SECONDS}s - possible DDP deadlock")
                want_stop = True # Force all ranks to stop together
                if _all_ranks_agree_skip(want_stop, self.device):
                    break
    
            want_stop    = (global_step > max_steps)
            dataset_name = "unknown"
            batch        = None

            if not want_stop:
                try:
                    dataset_name, batch = next(self.mixed_sampler)
                except StopIteration:
                    want_stop = True

            batch_is_none = (batch is None)
            agreed = self._rank_flags_agree({
                "stop":     want_stop,
                "no_batch": batch_is_none,
            })

            if agreed["stop"]:
                if self.is_main:
                    if global_step > max_steps:
                        logger.info(f"Reached max_steps ({max_steps}), all ranks stopping")
                    else:
                        logger.info("Sampler exhausted on at least one rank, all ranks stopping")
                break

            if agreed["no_batch"]:
                dataset_stats[dataset_name]["skipped"] += 1
                if self.is_main:
                    logger.warning(
                        f"Rank {self.rank}: None batch from {dataset_name} "
                        f"at step {global_step}, all ranks skipping"
                    )
                continue

            batch = self._to_device(batch)

            # Scheduled sampling probability
            _sched_enabled = self.cfg["train"].get("SCHED_SAMPLE_ENABLED", False)
            if _sched_enabled:
                _sched_start  = self.cfg["train"].get("SCHED_SAMPLE_START_STEP", 0)
                _sched_ramp   = self.cfg["train"].get("SCHED_SAMPLE_RAMP_STEPS", 200000)
                _sched_max_p  = self.cfg["train"].get("SCHED_SAMPLE_MAX_P", 0.40)
                _steps_in     = max(0, global_step - _sched_start)
                _p_sample     = min(_sched_max_p,
                                    _sched_max_p * _steps_in / max(_sched_ramp, 1))
            else:
                _p_sample = 0.0
            batch["_p_sample"] = _p_sample   # pass to loss_fn / model via batch
            #-------------------------------------------------------------------

            # FORWARD PASS
            loss     = None
            outputs  = {}
            losses   = {}
            fw_error = False
            try:
                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(**self._prepare_model_inputs(batch))
                    losses = self.loss_fn(outputs, batch)
                    loss = losses.get("total")

                    _xattn_ent_snapshot = outputs.get("_xattn_entropy")

                    # --- Scheduled sampling (full-Fv training) --------------
                    # After first teacher-forced forward, replace a fraction of
                    # input tokens with model's own argmax predictions, then
                    # run a second forward with these mixed inputs.
                    # Only fires on SAbDab batches with p_sample > 0.
                    _p_sample = batch.get("_p_sample", 0.0)
                    if (_p_sample > 0.001
                            and dataset_name == "sabdab"
                            and outputs.get("logits_ar") is not None
                            and loss is not None):
                        _lg = outputs["logits_ar"].detach()         # (B, L-1, V)
                        _model_preds = _lg.argmax(dim=-1)           # (B, L-1)
                        _orig_tokens = batch.get("seq_tokens_orig",
                                                 batch["seq_tokens"])
                        B2, L2 = _orig_tokens.shape

                        # Build mixed input: replace positions with prob p_sample
                        _mixed = _orig_tokens.clone()
                        _replace_mask = (
                            (torch.rand(B2, L2, device=_mixed.device) < _p_sample)
                            & batch["seq_mask"].bool()
                        )
                        _replace_mask[:, 0] = False   # never replace START

                        # Align model_preds (L-1) to token space (L)
                        _preds_full = torch.cat([
                            _orig_tokens[:, :1],   # pos 0 unchanged
                            _model_preds,
                        ], dim=1)[:, :L2]
                        _mixed[_replace_mask] = _preds_full[_replace_mask]

                        # Second forward with mixed tokens
                        _batch2 = dict(batch)
                        _batch2["seq_tokens"]      = _mixed
                        _batch2["seq_tokens_orig"] = _orig_tokens   # targets unchanged
                        _batch2["_p_sample"]       = 0.0            # no recursion

                        _out2    = self.model(**self._prepare_model_inputs(_batch2))
                        _loss2   = self.loss_fn(_out2, _batch2).get("total")

                        if _loss2 is not None and torch.isfinite(_loss2):
                            # Blend: alpha ramps from 0 to 0.5 as p_sample grows
                            _alpha = _p_sample / max(self.cfg["train"].get(
                                    "SCHED_SAMPLE_MAX_P", 0.40), 1e-6) * 0.5
                            loss = (1.0 - _alpha) * loss + _alpha * _loss2

                        # Log p_sample periodically
                        if global_step % 5000 == 0 and self.is_main:
                            print(
                                f"[ScheduledSampling step={global_step}] "
                                f"p_sample={_p_sample:.4f}  "
                                f"alpha={_alpha:.3f}  "
                                f"replaced={_replace_mask.float().mean():.3f}",
                                flush=True
                            )
                    #-----------------------------------------------------
                    # Apply gradient weight if specified
                    grad_weight = self.grad_weights.get(dataset_name, 1.0)
                    if grad_weight != 1.0 and loss is not None:
                        loss = loss * grad_weight
                        
                    # Anti-invariance loss
                    # Force CDR3 distribution to depend on antigen *identity*,
                    # not just antigen *presence*. Second forward with shuffled
                    # antigen runs under no_grad (only real-antigen path trains).

                    _ai_w = float(self.cfg.get("loss_weights", {}).get("ANTI_INV", 0.0))

                    if (_ai_w > 0
                            and dataset_name == "sabdab"
                            and batch.get("antigen_coords") is not None
                            and loss is not None):

                        _ag = batch["antigen_coords"]
                        _B = _ag.shape[0]

                        if _B > 1:

                            # Cross-class permutation: each sample gets an antigen
                            # from a DIFFERENT target class where possible.
                            # Falls back to roll if class labels are unavailable.
                            import random as _random
                            _targets = batch.get("targets")
                            _has_classes = (
                                _targets is not None
                                and len({str(t) for t in _targets if t is not None}) >= 2
                            )
                            if _has_classes:
                                _perm = list(range(_B))
                                for _i in range(_B):
                                    _cands = [
                                        _j for _j in range(_B)
                                        if _targets[_j] is not None
                                        and _targets[_j] != _targets[_i]
                                    ]
                                    if _cands:
                                        _perm[_i] = _random.choice(_cands)
                            else:
                                _perm = torch.roll(
                                    torch.arange(_B, device=_ag.device),
                                    max(1, _B // 2),
                                ).tolist()

                            _si = dict(self._prepare_model_inputs(batch))
                            _si["antigen_coords"] = _ag[_perm]

                            if batch.get("antigen_coord_mask") is not None:
                                _si["antigen_coord_mask"] = batch["antigen_coord_mask"][_perm]

                            if batch.get("antigen_seq_tokens") is not None:
                                _si["antigen_seq_tokens"] = batch["antigen_seq_tokens"][_perm]

                            with torch.no_grad():
                                _out_shuf = self.model(**_si)

                            _lr = outputs.get("logits_ar")
                            _ls = _out_shuf.get("logits_ar")

                            if _lr is not None and _ls is not None:

                                _c3 = batch.get("cdr3_mask")
                                _sm = batch.get("seq_mask")

                                if _c3 is not None and _sm is not None:

                                    _L = min(
                                        _lr.shape[1],
                                        _ls.shape[1],
                                        _c3.shape[1] - 1,
                                    )

                                    _val = (
                                        _sm[:, 1:][:, :_L].bool()
                                        & _c3[:, 1:][:, :_L].bool()
                                    )

                                    if _val.any():

                                        # ----------------------------------
                                        # KL anti-invariance loss
                                        # ----------------------------------

                                        _pr = _F.softmax(
                                            _lr[:, :_L][_val].float(),
                                            dim=-1
                                        ).clamp(min=1e-9)

                                        _ps = _F.softmax(
                                            _ls[:, :_L][_val].float().detach(),
                                            dim=-1
                                        ).clamp(min=1e-9)

                                        _kl = (
                                            _pr * (_pr.log() - _ps.log())
                                        ).sum(-1).mean()

                                        _margin = float(
                                            self.cfg.get("loss_weights", {})
                                            .get("ANTI_INV_MARGIN", 0.05)
                                        )

                                        _ai_loss = _F.relu(_margin - _kl)

                                        loss = loss + _ai_w * _ai_loss

                                        losses["anti_inv_kl"] = _kl.detach()
                                        losses["anti_inv_loss"] = _ai_loss.detach()
                                        losses["loss_anti_inv"] = _ai_loss.detach()

                                        # ----------------------------------
                                        # CDR3 NLL ranking loss
                                        # Real antigen should predict
                                        # true CDR3 residues better than
                                        # shuffled antigen.
                                        # ----------------------------------

                                        _cdr3_tgt = batch.get(
                                            "seq_tokens_orig",
                                            batch["seq_tokens"]
                                        )[:, 1:].contiguous()[:, :_L]

                                        _nll_real = _F.cross_entropy(
                                            _lr[:, :_L].reshape(
                                                -1,
                                                _lr.shape[-1]
                                            ),
                                            _cdr3_tgt.reshape(-1),
                                            ignore_index=0,
                                            reduction="none",
                                        ).reshape(_cdr3_tgt.shape)

                                        _nll_shuf = _F.cross_entropy(
                                            _ls[:, :_L].detach().reshape(
                                                -1,
                                                _ls.shape[-1]
                                            ),
                                            _cdr3_tgt.reshape(-1),
                                            ignore_index=0,
                                            reduction="none",
                                        ).reshape(_cdr3_tgt.shape)

                                        _nll_real_cdr3 = _nll_real[_val].mean()
                                        _nll_shuf_cdr3 = _nll_shuf[_val].mean()

                                        _nll_margin = float(
                                            self.cfg.get("loss_weights", {})
                                            .get("ANTI_INV_NLL_MARGIN", 0.30)
                                        )

                                        _nll_rank_loss = _F.relu(
                                            _nll_margin
                                            + _nll_real_cdr3
                                            - _nll_shuf_cdr3
                                        )

                                        _nll_rank_w = float(
                                            self.cfg.get("loss_weights", {})
                                            .get("ANTI_INV_NLL_WEIGHT", 0.10)
                                        )

                                        loss = loss + (
                                            _nll_rank_w * _nll_rank_loss
                                        )

                                        losses["anti_inv_nll_real"] = (
                                            _nll_real_cdr3.detach()
                                        )

                                        losses["anti_inv_nll_shuf"] = (
                                            _nll_shuf_cdr3.detach()
                                        )

                                        losses["anti_inv_nll_rank"] = (
                                            _nll_rank_loss.detach()
                                        )
                                        
                                        # InfoNCE: antigen embedding structure
                                        _infonce_w = float(
                                            self.cfg.get("loss_weights", {})
                                            .get("ANTI_INV_INFONCE_WEIGHT", 0.0)
                                        )
                                        _infonce_t = float(
                                            self.cfg.get("loss_weights", {})
                                            .get("ANTI_INV_INFONCE_TEMP", 0.10)
                                        )
                                        if _infonce_w > 0.0:
                                            _ag_pooled = outputs.get("antigen_pooled")
                                            if _ag_pooled is not None and _ag_pooled.shape[0] >= 2:
                                                from model.training_losses import antigen_infonce_loss
                                                _infonce = antigen_infonce_loss(
                                                    _ag_pooled, temperature=_infonce_t
                                                )
                                                loss = loss + _infonce_w * _infonce
                                                losses["anti_inv_infonce"] = _infonce.detach()

                                                # --- READ-ONLY DIAGNOSTIC: antigen_pooled collapse check ---
                                                # If all antigen embeddings in the batch have collapsed
                                                # toward the same direction, off-diagonal cosine sim -> 1.0
                                                # and this self-identity InfoNCE degenerates toward 0
                                                # regardless of whether antigen identity is preserved.
                                                # Zero grad impact: detached, no_grad, logging only.
                                                if global_step % 5000 == 0 and self.is_main:
                                                    with torch.no_grad():
                                                        _z = _F.normalize(_ag_pooled.float(), dim=-1)
                                                        _sim = _z @ _z.T  # (B, B)
                                                        _B = _sim.shape[0]
                                                        _off_diag = _sim[~torch.eye(_B, dtype=torch.bool,
                                                                                     device=_sim.device)]
                                                        print(
                                                            f"[ANTIGEN-POOLED-COLLAPSE step={global_step}] "
                                                            f"B={_B}  "
                                                            f"off_diag_cos_mean={_off_diag.mean().item():.4f}  "
                                                            f"off_diag_cos_std={_off_diag.std().item():.4f}  "
                                                            f"off_diag_cos_max={_off_diag.max().item():.4f}  "
                                                            f"norm_mean={_ag_pooled.norm(dim=-1).mean().item():.4f}  "
                                                            f"norm_std={_ag_pooled.norm(dim=-1).std().item():.4f}",
                                                            flush=True
                                                        )
            except Exception as e:
                fw_error = True
                if self.is_main:
                    logger.error(f"Error in {dataset_name} step {global_step}: {e}")
                    import traceback
                    traceback.print_exc()
            
            loss_bad = (
                fw_error
                or loss is None
                or not torch.isfinite(loss)
                or loss.grad_fn is None
            )
            if _all_ranks_agree_skip(loss_bad, self.device):
                if loss_bad:
                    dataset_stats[dataset_name]["skipped"] += 1
                    self._consecutive_failures[dataset_name] += 1
                    if self.is_main:
                        if fw_error:
                            logger.warning(
                                f"Forward error for {dataset_name} at step {global_step}"
                            )
                        elif loss is None:
                            logger.warning(
                                f"None loss for {dataset_name} at step {global_step}"
                            )
                        elif not torch.isfinite(loss):
                            try:
                                diag_parts = [f"dataset={dataset_name}"]
                                
                                if outputs.get('_seq_enc_nan'):
                                    diag_parts.append("SEQ_ENC_HAD_NAN=True")
                                else:
                                    # encoder was clean, NaN is in decoder
                                    if 'logits_ar' in outputs and outputs['logits_ar'] is not None:
                                        if not torch.isfinite(outputs['logits_ar']).all():
                                            diag_parts.append("NAN_SOURCE=decoder")
                                
                                if 'logits_ar' in outputs and outputs['logits_ar'] is not None:
                                    la = outputs['logits_ar']
                                    diag_parts.append(f"logits_ar_finite={torch.isfinite(la).all().item()}")
                                    diag_parts.append(f"logits_ar_absmax={la.abs().max().item():.2f}")
                                
                                if 'logits_mlm' in outputs and outputs['logits_mlm'] is not None:
                                    lm = outputs['logits_mlm']
                                    diag_parts.append(f"logits_mlm_finite={torch.isfinite(lm).all().item()}")
                                    diag_parts.append(f"logits_mlm_absmax={lm.abs().max().item():.2f}")
                                
                                if 'cond_repr' in outputs and outputs['cond_repr'] is not None:
                                    diag_parts.append(f"cond_finite={torch.isfinite(outputs['cond_repr']).all().item()}")
                                
                                # seq_enc is local to the model forward - expose it via outputs
                                if 'seq_enc' in outputs and outputs['seq_enc'] is not None:
                                    se = outputs['seq_enc']
                                    diag_parts.append(f"seq_enc_finite={torch.isfinite(se).all().item()}")
                                    diag_parts.append(f"seq_enc_absmax={se.abs().max().item():.2f}")
                                    diag_parts.append(f"seq_enc_norm_mean={se.norm(dim=-1).mean().item():.3f}")
                                
                                logger.warning(f"[NAN-ORIGIN step={global_step}] " + "  ".join(diag_parts))
                            except Exception as e:
                                logger.warning(f"[NAN-ORIGIN step={global_step}] tracer_error={e}")
                        else:
                            logger.warning(
                                f"Disconnected loss (no grad_fn) for "
                                f"{dataset_name} at step {global_step}"
                            )
                    if self.use_amp and self._consecutive_failures[dataset_name] > 5:
                        if self.is_main:
                            logger.warning(
                                f"Resetting grad scaler after repeated failures "
                                f"on {dataset_name}"
                            )
                        self.scaler = GradScaler(
                            'cuda', init_scale=1024, growth_factor=1.5,
                            backoff_factor=0.5, growth_interval=4000,
                        )
                        self._consecutive_failures.clear()
                    self.optimizer.zero_grad(set_to_none=True)
                continue
            else:
                self._consecutive_failures[dataset_name] = 0
            
            if loss > 10 and self.is_main:
                logger.warning(f"Large loss detected: {loss.item():.4f}")
            
            # BACKWARD PASS & OPTIMIZER STEP
            self.optimizer.zero_grad(set_to_none=True)       
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            _m = self.model.module if hasattr(self.model, 'module') else self.model
            gate_clamp      = self.cfg["train"].get("gate_logit_clamp", 3.0)
            if hasattr(_m, 'geom_cond_gate') and _m.geom_cond_gate is not None:
                with torch.no_grad():
                    _m.geom_cond_gate.clamp_(-gate_clamp, gate_clamp)
            if hasattr(_m, 'antigen_xattn_gate'):
                with torch.no_grad():
                    _m.antigen_xattn_gate.clamp_(-3.0, 3.0)
            
            def fmt(x):
                return f"{x:.3f}" if x is not None else "None"

            offset = global_step - resume_step
            if offset in DEBUG_OFFSETS and self.is_main:
                geom_logit = _m.geom_cond_gate.item() if hasattr(_m, 'geom_cond_gate') and _m.geom_cond_gate is not None else None
                ag_logit   = _m.antigen_xattn_gate.item() if hasattr(_m, 'antigen_xattn_gate') else None
                print(
                    f"[GATE-CLAMP step={global_step}] "
                    f"geom_logit={fmt(geom_logit)} "
                    f"ag_logit={fmt(ag_logit)}",
                    flush=True
                )
                print(f"[{datetime.now()}] [STEP {global_step}] Backward pass done | Dataset: {dataset_name}, loss={loss.item():.4f} | time spent: {time.time() - last_successful_step_time:.2f}s", flush=True)
                if "anti_inv_kl" in losses:
                    print(f"[ANTI-INV-INSTANT step={global_step}] "
                        f"kl={losses['anti_inv_kl'].item():.4f}  "
                        f"loss={losses.get('anti_inv_loss', torch.tensor(0)).item():.6f}  "
                        f"nll_rank={losses.get('anti_inv_nll_rank', torch.tensor(0)).item():.4f}",
                        flush=True)
                
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                
            last_successful_step_time = time.time()
            
            # Log LR periodically
            if self.scheduler is not None and global_step % self.log_interval == 0 and self.is_main:
                print("=" * 60, flush=True)
                print(f"\nLearning rate at optimizer step {global_step}: {self.scheduler.get_last_lr()[0]:.2e}",
                            flush=True)
            
            # Log anti-invariance losses periodically and gates
            ai_cnt = max(overall_stats["anti_inv_count"], 1)
            avg_ai_loss = overall_stats["anti_inv"] / ai_cnt
            avg_ai_kl = overall_stats["anti_inv_kl"] / ai_cnt
            avg_ai_nll_real = overall_stats["anti_inv_nll_real"] / ai_cnt
            avg_ai_nll_shuf = overall_stats["anti_inv_nll_shuf"] / ai_cnt
            avg_ai_nll_rank = overall_stats["anti_inv_nll_rank"] / ai_cnt
            if global_step % 5000 == 0 and self.is_main:
                _m = self.model.module if hasattr(self.model, 'module') else self.model
                geom_logit = _m.geom_cond_gate.item() if (hasattr(_m, 'geom_cond_gate') and _m.geom_cond_gate is not None) else None
                ag_logit   = _m.antigen_xattn_gate.item() if hasattr(_m, 'antigen_xattn_gate') else None
                sig = lambda x: torch.sigmoid(torch.tensor(x)).item() if x is not None else None
                geom_sig = sig(geom_logit)
                if ag_logit is not None:
                    ag_sig = torch.sigmoid(torch.tensor(ag_logit).clamp(-3.0, 3.0)).item() * 0.75 + 0.05
                else:
                    ag_sig = None
                parts = []
                lparts = []

                if overall_stats["anti_inv_count"] > 0:
                    lparts.append(f"anti_inv_loss={avg_ai_loss:.4f}")
                    lparts.append(f"anti_inv_kl={avg_ai_kl:.4f}")
                    lparts.append(f"anti_inv_nll_real={avg_ai_nll_real:.4f}")
                    lparts.append(f"anti_inv_nll_shuf={avg_ai_nll_shuf:.4f}")
                    lparts.append(f"anti_inv_nll_rank={avg_ai_nll_rank:.4f}")
                if overall_stats.get("anti_inv_infonce", 0.0) > 0:
                    avg_infonce = overall_stats["anti_inv_infonce"] / max(overall_stats["anti_inv_count"], 1)
                    lparts.append(f"anti_inv_infonce={avg_infonce:.4f}")
                print(f"[AI losses step={global_step}] " + "  ".join(lparts), flush=True)
                # Scheduled sampling status
                if _p_sample > 0.0:
                    print(
                        f"[ScheduledSampling step={global_step}] "
                        f"p_sample={_p_sample:.4f}  "
                        f"(ramp: {_p_sample / max(self.cfg.get('SCHED_SAMPLE_MAX_P', 0.40), 1e-6) * 100:.1f}% of max)",
                        flush=True
                    )
                if geom_sig is not None: parts.append(f"geom={geom_sig:.3f}(logit={geom_logit:.3f})")
                if ag_sig   is not None: parts.append(f"ag_xattn={ag_sig:.3f}(logit={ag_logit:.3f})")
                print(f"[GATE step={global_step}] " + "  ".join(parts), flush=True)

                # Read-only attention-weight entropy/sharpness, from the
                # PRIMARY forward pass snapshot taken earlier this step.
                # ratio -> 1.0 = diffuse/uninformative attention (supports
                # the §5.4 noise-injection mechanism hypothesis).
                # ratio -> 0.0 = sharp/peaked attention (mechanism resolving
                # naturally with more training, no architecture change needed).
                _ent_snap = locals().get("_xattn_ent_snapshot")
                if _ent_snap:
                    ent_parts = []
                    for _site_key in (2, 4, "post"):
                        if _site_key in _ent_snap:
                            _mean_ent, _max_ent = _ent_snap[_site_key]
                            _ratio = (_mean_ent / _max_ent.clamp(min=1e-9)).item()
                            ent_parts.append(
                                f"L{_site_key}_ratio={_ratio:.3f}"
                                f"(ent={_mean_ent.item():.3f}/{_max_ent.item():.3f})"
                            )
                    if ent_parts:
                        print(f"[XATTN-ENTROPY step={global_step}] " + "  ".join(ent_parts), flush=True)
            
            # UPDATE STATS
            grad_weight = self.grad_weights.get(dataset_name, 1.0)
            loss_val = loss.item() / grad_weight  # Normalize for logging
            dataset_stats[dataset_name]["total"] += loss_val
            dataset_stats[dataset_name]["count"] += 1
            overall_stats["total"] += loss_val
    
            for key in ["ar", "mlm", "geo", "if", "anti_inv"]:
                lkey = f"loss_{key}"
                if lkey in losses and losses[lkey] is not None:
                    component_loss = losses[lkey].item()
                    dataset_stats[dataset_name][key] += component_loss
                    overall_stats[key] += component_loss
            if "anti_inv_kl" in losses:
                v = losses["anti_inv_kl"].item()
                dataset_stats[dataset_name]["anti_inv_kl"] += v
                overall_stats["anti_inv_kl"] += v

            if "anti_inv_nll_real" in losses:
                v = losses["anti_inv_nll_real"].item()
                dataset_stats[dataset_name]["anti_inv_nll_real"] += v
                overall_stats["anti_inv_nll_real"] += v

            if "anti_inv_nll_shuf" in losses:
                v = losses["anti_inv_nll_shuf"].item()
                dataset_stats[dataset_name]["anti_inv_nll_shuf"] += v
                overall_stats["anti_inv_nll_shuf"] += v

            if "anti_inv_nll_rank" in losses:
                v = losses["anti_inv_nll_rank"].item()
                dataset_stats[dataset_name]["anti_inv_nll_rank"] += v
                overall_stats["anti_inv_nll_rank"] += v

            if "anti_inv_infonce" in losses:
                v = losses["anti_inv_infonce"].item()
                dataset_stats[dataset_name]["anti_inv_infonce"] += v
                overall_stats["anti_inv_infonce"] += v
            
            if "anti_inv_loss" in losses:
                dataset_stats[dataset_name]["anti_inv_count"] += 1
                overall_stats["anti_inv_count"] += 1
            
            
            # AR accuracy
            task_mask = batch.get("task_mask", {})
            real_mask = ~batch["synthetic_negative_mask"]
            if task_mask.get("AR") and "logits_ar" in outputs:
            
                valid_rows = (
                    batch["sample_task_mask"]["AR"]
                    & real_mask
                )

                if valid_rows.sum() == 0:
                    pass
                else:
                    logits = outputs["logits_ar"].detach()[valid_rows]
                    _ar_tokens = batch.get("seq_tokens_orig")
                    if _ar_tokens is None:
                        _ar_tokens = batch["seq_tokens"]
                    tokens = _ar_tokens[valid_rows]
                    seq_mask = batch["seq_mask"][valid_rows]

                    cdr_mask = batch.get("cdr_mask")
                    if cdr_mask is not None:
                        cdr_mask = cdr_mask[valid_rows]
                
                    # shift targets
                    targets = tokens[:, 1:].contiguous()
                    seq_mask = seq_mask[:, 1:].contiguous()
                
                    if cdr_mask is not None:
                        cdr_mask = cdr_mask[:, 1:].contiguous()
                
                    # align length
                    min_len = min(logits.shape[1], targets.shape[1])
                    logits = logits[:, :min_len]
                    targets = targets[:, :min_len]
                    seq_mask = seq_mask[:, :min_len]
                
                    if cdr_mask is not None:
                        cdr_mask = cdr_mask[:, :min_len]
                
                    preds = logits.argmax(dim=-1)
                    
                    seq_mask = seq_mask * (targets != 4)
                    
                    if global_step % self.log_interval == 0 and self.is_main:
                        lgt = logits.float()
                        print(f"[DECODER-DIAG step={global_step} ds={dataset_name}]", flush=True)
                        print(f"  logits mean={lgt.mean().item():.4f}  std={lgt.std().item():.4f}", flush=True)
                        tok_dist = torch.bincount(preds.flatten(), minlength=25).tolist()
                        tgt_dist = torch.bincount(targets.flatten(), minlength=25).tolist()
                        print(f"  preds  dist (tok 0-24): {tok_dist}", flush=True)
                        print(f"  target dist (tok 0-24): {tgt_dist}", flush=True)
                        
                    # masks (same as loss)
                    if cdr_mask is not None:
                        fw_mask = seq_mask * (1 - cdr_mask.float())
                        cdr_weight_mask = seq_mask * cdr_mask.float()
                    else:
                        fw_mask = seq_mask
                        cdr_weight_mask = torch.zeros_like(seq_mask)
                    
                    
                
                    # correct counts
                    fw_correct = ((preds == targets) * fw_mask.bool()).sum().item()
                    fw_total   = fw_mask.sum().item()
                
                    cdr_correct = ((preds == targets) * cdr_weight_mask.bool()).sum().item()
                    cdr_total   = cdr_weight_mask.sum().item()
                
                    # store
                    dataset_stats[dataset_name]["fw_correct"] += fw_correct
                    dataset_stats[dataset_name]["fw_total"]   += fw_total
                    dataset_stats[dataset_name]["cdr_correct"] += cdr_correct
                    dataset_stats[dataset_name]["cdr_total"]   += cdr_total
                
                    overall_stats["fw_correct"] += fw_correct
                    overall_stats["fw_total"]   += fw_total
                    overall_stats["cdr_correct"] += cdr_correct
                    overall_stats["cdr_total"]   += cdr_total

                    # CDR3-only accuracy
                    cdr3_mask_raw = batch.get("cdr3_mask")
                    if cdr3_mask_raw is not None:
                        cdr3_mask_raw = cdr3_mask_raw[valid_rows]
                        cdr3_m = cdr3_mask_raw[:, 1:].contiguous()[:, :min_len]
                        cdr3_weighted = seq_mask * cdr3_m.float()

                        # Effective vocabulary on CDR3 positions
                        if global_step % self.log_interval == 0 and self.is_main:
                            _cdr3_preds = preds[cdr3_weighted.bool()]
                            if _cdr3_preds.numel() > 0:
                                _counts = torch.bincount(
                                    _cdr3_preds.clamp(0, 24), minlength=25
                                ).float()[5:25]
                                _p = _counts / _counts.sum().clamp(min=1e-9)
                                _p = _p[_p > 0]
                                _ev_pred = float(torch.exp(-(_p * _p.log()).sum()))

                                _counts_t = torch.bincount(
                                    targets[cdr3_weighted.bool()].clamp(0, 24),
                                    minlength=25
                                ).float()[5:25]
                                _p_t = _counts_t / _counts_t.sum().clamp(min=1e-9)
                                _p_t = _p_t[_p_t > 0]
                                _ev_tgt = float(torch.exp(-(_p_t * _p_t.log()).sum()))

                                print(
                                    f"  EV_pred={_ev_pred:.2f}  EV_target={_ev_tgt:.2f}  "
                                    f"EV_ratio={_ev_pred/_ev_tgt:.3f}  "
                                    f"(healthy: ratio>0.5, EV_pred>8)",
                                    flush=True
                                )

                        cdr3_correct = ((preds == targets) * cdr3_weighted.bool()).sum().item()
        
                        cdr3_correct = ((preds == targets) * cdr3_weighted.bool()).sum().item()
                        cdr3_total   = cdr3_weighted.sum().item()
        
                        dataset_stats[dataset_name]["cdr3_correct"] += cdr3_correct
                        dataset_stats[dataset_name]["cdr3_total"]   += cdr3_total
                        overall_stats["cdr3_correct"] += cdr3_correct
                        overall_stats["cdr3_total"]   += cdr3_total
                        
                        # Per-chain CDR3 split (H3 vs L3), same shifted/aligned
                        # masks as the loss weighting. cdr3_weighted already
                        # zeroes pad/MASK, so chain garbage at pads can't leak in.
                        chain_raw = batch.get("chain_position_mask")
                        if chain_raw is not None:
                            chain_raw = chain_raw[valid_rows]
                            ch = chain_raw[:, 1:].contiguous()[:, :min_len].float()
                            is_heavy = (ch < 0.5).float()
                            is_light = 1.0 - is_heavy          # complement: no double-count
                            h3_w = cdr3_weighted * is_heavy
                            l3_w = cdr3_weighted * is_light

                            h3_correct = ((preds == targets) * h3_w.bool()).sum().item()
                            h3_total   = h3_w.sum().item()
                            l3_correct = ((preds == targets) * l3_w.bool()).sum().item()
                            l3_total   = l3_w.sum().item()

                            dataset_stats[dataset_name]["h3_correct"] += h3_correct
                            dataset_stats[dataset_name]["h3_total"]   += h3_total
                            dataset_stats[dataset_name]["l3_correct"] += l3_correct
                            dataset_stats[dataset_name]["l3_total"]   += l3_total
                            overall_stats["h3_correct"] += h3_correct
                            overall_stats["h3_total"]   += h3_total
                            overall_stats["l3_correct"] += l3_correct
                            overall_stats["l3_total"]   += l3_total

                            # Contact-vs-noncontact split within H3 only -
                            # h3_w is already H3-positions-weighted-by-validity;
                            # contact_mask is H3-only by construction (loader-side),
                            # so intersecting is a clean subset, no double-counting.
                            contact_raw = batch.get("contact_mask")
                            if contact_raw is not None:
                                contact_raw = contact_raw[valid_rows]
                                co = contact_raw[:, 1:].contiguous()[:, :min_len].float()
                                h3_contact_w    = h3_w * co
                                h3_noncontact_w = h3_w * (1.0 - co)
 
                                h3c_correct = ((preds == targets) * h3_contact_w.bool()).sum().item()
                                h3c_total   = h3_contact_w.sum().item()
                                h3n_correct = ((preds == targets) * h3_noncontact_w.bool()).sum().item()
                                h3n_total   = h3_noncontact_w.sum().item()
 
                                dataset_stats[dataset_name]["h3_contact_correct"]    += h3c_correct
                                dataset_stats[dataset_name]["h3_contact_total"]      += h3c_total
                                dataset_stats[dataset_name]["h3_noncontact_correct"] += h3n_correct
                                dataset_stats[dataset_name]["h3_noncontact_total"]   += h3n_total
                                overall_stats["h3_contact_correct"]    += h3c_correct
                                overall_stats["h3_contact_total"]      += h3c_total
                                overall_stats["h3_noncontact_correct"] += h3n_correct
                                overall_stats["h3_noncontact_total"]   += h3n_total
 
                                if global_step % self.log_interval == 0 and self.is_main:
                                    print(
                                        f"  H3-contact-split [ds={dataset_name}]: "
                                        f"contact_acc={h3c_correct / max(h3c_total, 1):.3f} "
                                        f"(n={int(h3c_total)})  "
                                        f"noncontact_acc={h3n_correct / max(h3n_total, 1):.3f} "
                                        f"(n={int(h3n_total)})",
                                        flush=True,
                                    )
                                del co, h3_contact_w, h3_noncontact_w

                            if global_step % self.log_interval == 0 and self.is_main:
                                print(
                                    f"  CDR3-by-chain [ds={dataset_name}]: "
                                    f"H3_acc={h3_correct / max(h3_total, 1):.3f} "
                                    f"(n={int(h3_total)})  "
                                    f"L3_acc={l3_correct / max(l3_total, 1):.3f} "
                                    f"(n={int(l3_total)})",
                                    flush=True,
                                )
                            del ch, is_heavy, is_light, h3_w, l3_w
                        del cdr3_m, cdr3_weighted
                    del logits
    
            # MLM accuracy
            if task_mask.get("MLM") and "logits_mlm" in outputs:
            
                valid_rows = (
                    batch["sample_task_mask"]["MLM"]
                    & real_mask
                )

                if valid_rows.sum() == 0:
                    pass
                else:    
                    mlm_logits = outputs["logits_mlm"].detach()[valid_rows]

                    mlm_labels = batch.get("mlm_labels")
                    if mlm_labels is not None:
                        mlm_labels = mlm_labels[valid_rows]
                
                    if mlm_labels is not None:
                        cdr_mask_mlm = batch.get("cdr_mask")

                        if cdr_mask_mlm is not None:
                            cdr_mask_mlm = cdr_mask_mlm[valid_rows]
                
                        # align shapes
                        min_len = min(mlm_logits.shape[1], mlm_labels.shape[1])
                        mlm_logits = mlm_logits[:, :min_len]
                        mlm_labels = mlm_labels[:, :min_len]
                
                        if cdr_mask_mlm is not None:
                            cdr_mask_mlm = cdr_mask_mlm[:, :min_len]
                
                        mlm_mask = mlm_labels != -100
                
                        preds = mlm_logits.argmax(dim=-1)
                
                        correct = ((preds == mlm_labels) * mlm_mask).sum().item()
                        total   = mlm_mask.sum().item()
                
                        dataset_stats[dataset_name]["mlm_correct"] += correct
                        dataset_stats[dataset_name]["mlm_total"]   += total
                        overall_stats["mlm_correct"] += correct
                        overall_stats["mlm_total"]   += total
                    del mlm_logits
    
            # CLEANUP MEMORY        
            if global_step % 2000 == 0:
                self._cleanup_memory()
    
            # STEP-BASED VALIDATION        
            if val_check_interval and global_step % val_check_interval == 0:
                if self.is_main:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[Step {global_step}] Running validation...")
                    logger.info("="*60)

                val_metrics = self.validate(epoch=epoch)

                if self.is_main and ckpt_dir:
                    self._save_best_checkpoint(val_metrics, epoch, global_step, best_val_loss, ckpt_dir)

                self._cleanup_memory()
                self.model.train()
    
            # STEP-BASED CHECKPOINTING        
            if save_every_n_steps and global_step % save_every_n_steps == 0:
                if self.is_main and ckpt_dir:
                    self._save_step_checkpoint(epoch, global_step, best_val_loss, ckpt_dir)
                    self._cleanup_memory()

            # LOGGING            
            total_batches = sum(ds["count"] for ds in dataset_stats.values())
            
            if global_step % self.log_interval == 0 and self.is_main and total_batches > 0:
                avg_loss = overall_stats["total"] / total_batches
                
                _m = self.model.module if hasattr(self.model, 'module') else self.model
                geom_gate_beta  = torch.sigmoid(_m.geom_cond_gate).item() \
                                if (hasattr(_m, 'geom_cond_gate')
                                    and _m.geom_cond_gate is not None) else None
                if hasattr(_m, 'antigen_xattn_gate'):
                    _raw = _m.antigen_xattn_gate.item()
                    antigen_gate_ag = torch.sigmoid(
                        torch.tensor(_raw).clamp(-3.0, 3.0)
                    ).item() * 0.75 + 0.05
                else:
                    antigen_gate_ag = None
            
                gate_str = ""
                if geom_gate_beta is not None:
                    gate_str += f"  geom_gate_b={geom_gate_beta:.3f}"
                if antigen_gate_ag is not None:
                    gate_str += f"  antigen_gate_y={antigen_gate_ag:.3f}"
                print(f"[Step {global_step}] Dataset used {dataset_name} | stats:", flush=True)
                print(
                    f"[{datetime.now()}] step={global_step} "
                    f"loss={avg_loss:.4f}-{gate_str}",
                    flush=True
                )
                print("=" * 60, flush=True)
            
                # Step-level wandb logging
                try:
                    import wandb as _wandb
                    if _wandb.run is not None and global_step % (self.log_interval * 5) == 0:
                        step_log = {"train/step_loss": avg_loss}
                        if geom_gate_beta is not None:
                            step_log["gates/geom_gate_beta"] = geom_gate_beta
                        if antigen_gate_ag is not None:
                            step_log["gates/antigen_gate_gamma"] = antigen_gate_ag
                        _wandb.log(step_log, step=global_step)
                except ImportError:
                    pass
    
            # Update progress bar
            if (step_in_epoch + 1) % self.log_interval == 0:
                if total_batches > 0:
                    avg_loss = overall_stats["total"] / total_batches
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
        # COMPUTE METRICS    
        dataset_metrics = self._compute_dataset_metrics(dataset_stats)
        overall_metrics = self._compute_overall_metrics(dataset_stats, overall_stats)
    
        if self.is_main:
            self._log_batch_summary(dataset_stats)
            self._log_training_metrics(dataset_metrics, overall_metrics)
            
        # LR scheduler Log
        if self.scheduler is not None:
            if self.is_main:
                logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}")
                
        return dataset_metrics, overall_metrics, global_step

    def _rank_flags_agree(self, flags: dict) -> dict:
        """
        Single all_reduce for all skip flags at once.
        flags: {name: bool} - local rank's votes
        Returns: {name: bool} - True if ANY rank voted True (MAX reduce)
        """
        if not (dist.is_available() and dist.is_initialized()):
            return flags
        
        keys = list(flags.keys())
        t = torch.tensor(
            [1.0 if flags[k] else 0.0 for k in keys],
            device=self.device
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return {k: t[i].item() > 0.0 for i, k in enumerate(keys)}

    def set_active_datasets(self, dataset_names):
        """Restrict validation to only these datasets."""
        self.active_val_loaders = {
            k: v for k, v in self.val_loaders.items() if k in dataset_names
        }
    
    def validate(self, epoch=None):
        all_metrics = {}
        loaders = getattr(self, "active_val_loaders", self.val_loaders)
        global_max = self.cfg["train"].get("max_val_batches", None)

        _was_training = self.model.training
        self.model.eval()
        try:
            for ds_name, loader in loaders.items():
                max_batches = self.cfg["train"].get(
                    f"max_val_batches_{ds_name}", global_max
                )
                metrics = validate(
                    self.model, loader, self.device,
                    vocab_size=self.cfg["model"]["vocab_size"],
                    max_batches=max_batches,
                )
                all_metrics[ds_name] = metrics
            if self.is_main:
                self._log_validation_metrics(all_metrics)
        finally:
            if _was_training:
                self.model.train()
        return all_metrics

    def _cleanup_memory(self):
        """Aggressive memory cleanup to prevent leaks."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _compute_dataset_metrics(self, dataset_stats):
        """Compute metrics for each dataset."""
        dataset_metrics = {}
        
        for ds_name, stats in dataset_stats.items():
            if stats["count"] > 0:
                metrics = {
                    "total_loss": stats["total"] / stats["count"],
                    "ar_loss": stats["ar"] / stats["count"] if stats["ar"] > 0 else 0.0,
                    "mlm_loss": stats["mlm"] / stats["count"] if stats["mlm"] > 0 else 0.0,
                    "geo_loss": stats["geo"] / stats["count"] if stats["geo"] > 0 else 0.0,
                    "if_loss": stats["if"] / stats["count"] if stats["if"] > 0 else 0.0,
                    "num_batches": stats["count"],
                    "anti_inv_loss": (
                                    stats["anti_inv"] / stats["count"]
                                    if stats["anti_inv"] > 0 else 0.0
                                ),
                }
                if stats["anti_inv_count"] > 0:
                    metrics["anti_inv_kl"] = (
                        stats["anti_inv_kl"] / stats["anti_inv_count"]
                    )

                    metrics["anti_inv_nll_real"] = (
                        stats["anti_inv_nll_real"] / stats["anti_inv_count"]
                    )

                    metrics["anti_inv_nll_shuf"] = (
                        stats["anti_inv_nll_shuf"] / stats["anti_inv_count"]
                    )

                    metrics["anti_inv_nll_rank"] = (
                        stats["anti_inv_nll_rank"] / stats["anti_inv_count"]
                    )
                if stats["anti_inv_infonce"] > 0:
                    metrics["anti_inv_infonce"] = (
                        stats["anti_inv_infonce"] / stats["anti_inv_count"]
                    )
                
                if stats["fw_total"] > 0:
                    metrics["fw_acc"] = stats["fw_correct"] / stats["fw_total"]
                if stats["cdr_total"] > 0:
                    metrics["cdr_acc"] = stats["cdr_correct"] / stats["cdr_total"]
                # CDR3-only accuracy (most therapeutically relevant)
                if stats["cdr3_total"] > 0:
                    metrics["cdr3_acc"] = stats["cdr3_correct"] / stats["cdr3_total"]
                if stats["h3_total"] > 0:
                    metrics["h3_acc"] = stats["h3_correct"] / stats["h3_total"]
                if stats["l3_total"] > 0:
                    metrics["l3_acc"] = stats["l3_correct"] / stats["l3_total"]
                if stats["h3_contact_total"] > 0:
                    metrics["h3_contact_acc"] = stats["h3_contact_correct"] / stats["h3_contact_total"]
                if stats["h3_noncontact_total"] > 0:
                    metrics["h3_noncontact_acc"] = stats["h3_noncontact_correct"] / stats["h3_noncontact_total"]
                total_tokens = stats.get("fw_total", 0) + stats.get("cdr_total", 0)
                total_correct = stats.get("fw_correct", 0) + stats.get("cdr_correct", 0)
                if total_tokens > 0:
                    metrics["ar_acc"] = total_correct / total_tokens
                if stats["mlm_total"] > 0:
                    metrics["mlm_acc"] = stats["mlm_correct"] / stats["mlm_total"]
                
                dataset_metrics[ds_name] = metrics
        return dataset_metrics
    
    def _compute_overall_metrics(self, dataset_stats, overall_stats):
        """Compute overall metrics across all datasets."""
        total_batches = sum(stats["count"] for stats in dataset_stats.values())
        if total_batches == 0:
            if self.is_main:
                logger.error("No batches processed in entire epoch!")
            return {"total_loss": 0.0, "ar_loss": 0.0, "mlm_loss": 0.0,
                    "geo_loss": 0.0, "if_loss": 0.0}
        
        overall_metrics = {
            "total_loss": overall_stats["total"] / total_batches,
            "ar_loss": overall_stats["ar"] / total_batches if overall_stats["ar"] > 0 else 0.0,
            "mlm_loss": overall_stats["mlm"] / total_batches if overall_stats["mlm"] > 0 else 0.0,
            "geo_loss": overall_stats["geo"] / total_batches if overall_stats["geo"] > 0 else 0.0,
            "if_loss": overall_stats["if"] / total_batches if overall_stats["if"] > 0 else 0.0,
            "anti_inv_loss": (
                            overall_stats["anti_inv"] / total_batches
                            if overall_stats["anti_inv"] > 0 else 0.0
                        ),
        }
        if overall_stats["anti_inv_count"] > 0:
            overall_metrics["anti_inv_kl"] = (
                overall_stats["anti_inv_kl"]
                / overall_stats["anti_inv_count"]
            )

            overall_metrics["anti_inv_nll_real"] = (
                overall_stats["anti_inv_nll_real"]
                / overall_stats["anti_inv_count"]
            )

            overall_metrics["anti_inv_nll_shuf"] = (
                overall_stats["anti_inv_nll_shuf"]
                / overall_stats["anti_inv_count"]
            )

            overall_metrics["anti_inv_nll_rank"] = (
                overall_stats["anti_inv_nll_rank"]
                / overall_stats["anti_inv_count"]
            )
        if overall_stats["anti_inv_infonce"] > 0:
            overall_metrics["anti_inv_infonce"] = (
                overall_stats["anti_inv_infonce"]
                / overall_stats["anti_inv_count"]
            )
        
        if overall_stats["fw_total"] > 0:
            overall_metrics["fw_acc"] = overall_stats["fw_correct"] / overall_stats["fw_total"]
        if overall_stats["cdr_total"] > 0:
            overall_metrics["cdr_acc"] = overall_stats["cdr_correct"] / overall_stats["cdr_total"]
        # CDR3-only accuracy
        if overall_stats["cdr3_total"] > 0:
            overall_metrics["cdr3_acc"] = overall_stats["cdr3_correct"] / overall_stats["cdr3_total"]
        if overall_stats["h3_total"] > 0:
            overall_metrics["h3_acc"] = overall_stats["h3_correct"] / overall_stats["h3_total"]
        if overall_stats["l3_total"] > 0:
            overall_metrics["l3_acc"] = overall_stats["l3_correct"] / overall_stats["l3_total"]
        if overall_stats["h3_contact_total"] > 0:
            overall_metrics["h3_contact_acc"] = overall_stats["h3_contact_correct"] / overall_stats["h3_contact_total"]
        if overall_stats["h3_noncontact_total"] > 0:
            overall_metrics["h3_noncontact_acc"] = overall_stats["h3_noncontact_correct"] / overall_stats["h3_noncontact_total"]
        total_tokens = overall_stats.get("fw_total", 0) + overall_stats.get("cdr_total", 0)
        total_correct = overall_stats.get("fw_correct", 0) + overall_stats.get("cdr_correct", 0)
        if total_tokens > 0:
            overall_metrics["ar_acc"] = total_correct / total_tokens
        if overall_stats["mlm_total"] > 0:
            overall_metrics["mlm_acc"] = overall_stats["mlm_correct"] / overall_stats["mlm_total"]

        
        return overall_metrics
    
    def _save_best_checkpoint(self, val_metrics, epoch, global_step, best_val_loss, ckpt_dir):
        """Save best checkpoint based on validation metrics."""
        from run_train import save_checkpoint
        
        # Flatten metrics
        val_metrics_flat = {}
        for ds_name, metrics in val_metrics.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v == v:
                    val_metrics_flat[f"{ds_name}_{k}"] = v
        
        # Check AR perplexity (primary metric)
        # Best-checkpoint metric priority -- mirrors run_train.py's
        # _BEST_MODEL_METRIC_PRIORITY so the step-based and epoch-based
        # validation paths can never silently disagree about what "best"
        # means. If this list is ever changed in run_train.py, change it
        # here too.
        _step_best_priority = [
            "sabdab_h3_ppl",
            "oas_hl_h3_ppl",
            "oas_paired_h3_ppl",
            "sabdab_cdr3_ppl",
            "oas_hl_cdr3_ppl",
            "oas_paired_cdr3_ppl",
            "oas_hl_ar_ppl",
            "oas_paired_ar_ppl",
            "sabdab_ar_ppl",
        ]
        score_key = None
        current_score = None
        for _key in _step_best_priority:
            if _key in val_metrics_flat:
                score_key = _key
                current_score = val_metrics_flat[_key]
                break

        if current_score is not None and current_score < best_val_loss:
            best_path = os.path.join(ckpt_dir, "scat_best.pt")
            save_checkpoint(
                best_path, epoch, global_step, self.model,
                self.optimizer, self.scheduler, current_score,
                val_metrics, self.cfg, sampler=self.mixed_sampler, scaler=self.scaler
            )
            if self.is_main:
                logger.info(f"New best model saved | {score_key}={current_score:.4f}")
        
        # Save best interface model
        interface_auc = (val_metrics_flat.get("sabdab_residue_iface_roc_auc") or 0.0)
        
        if interface_auc > 0.80:
            interface_path = os.path.join(ckpt_dir, "scat_interface_best.pt")
            
            should_save = True
            if os.path.exists(interface_path):
                try:
                    prev = torch.load(interface_path, map_location='cpu', weights_only=False)
                    prev_auc = max(
                        m.get('residue_iface_roc_auc', 0.0) or m.get('iface_roc_auc', 0.0)
                        for m in prev.get('val_metrics', {}).values()
                    )
                    should_save = (interface_auc > prev_auc)
                except:
                    pass
            
            if should_save:
                save_checkpoint(
                    interface_path, epoch, global_step, self.model,
                    self.optimizer, self.scheduler, best_val_loss,
                    val_metrics, self.cfg, sampler=self.mixed_sampler, scaler=self.scaler,
                )
                if self.is_main:
                    logger.info(f"Best interface model saved (AUC={interface_auc:.4f})")
    
    def _save_step_checkpoint(self, epoch, global_step, best_val_loss, ckpt_dir):
        """Save step-based checkpoint."""
        from run_train import save_checkpoint
        
        ckpt_path = os.path.join(ckpt_dir, f"scat_step_{global_step}.pt")
        save_checkpoint(
            ckpt_path, epoch, global_step, self.model,
            self.optimizer, self.scheduler, best_val_loss,
            {}, self.cfg, sampler=self.mixed_sampler, scaler=self.scaler
        )
        if self.is_main:
            logger.info(f"Checkpoint saved at step {global_step}")
    
    def _log_batch_summary(self, dataset_stats):
        """Log batch processing summary."""
        if self.is_main:
            logger.info("\nDataset Processing Summary:")
            for ds_name, stats in dataset_stats.items():
                logger.info(f"  [{ds_name}] processed={stats['count']}, "
                       f"skipped={stats['skipped']}, errors={stats['errors']}")

    def _log_training_metrics(self, dataset_metrics, overall_metrics):
        """Log training metrics."""
        if self.is_main:
            logger.info("\nTraining Metrics:")
            for ds_name, metrics in dataset_metrics.items():
                parts = []
                
                if "total_loss" in metrics:
                    parts.append(f"loss={metrics['total_loss']:.4f}")
                
                if metrics.get('ar_loss', 0) > 0:
                    parts.append(f"ar_loss={metrics['ar_loss']:.4f}")
                if metrics.get('mlm_loss', 0) > 0:
                    parts.append(f"mlm_loss={metrics['mlm_loss']:.4f}")
                if metrics.get('geo_loss', 0) > 0:
                    parts.append(f"geo_loss={metrics['geo_loss']:.4f}")
                if metrics.get('if_loss', 0) > 0:
                    parts.append(f"if_loss={metrics['if_loss']:.4f}")
                if metrics.get('anti_inv_loss', 0) > 0:
                    parts.append(f"anti_inv_loss={metrics['anti_inv_loss']:.4f}")
                if "anti_inv_kl" in metrics:
                    parts.append(f"anti_inv_kl={metrics['anti_inv_kl']:.4f}")

                if "anti_inv_nll_real" in metrics:
                    parts.append(f"anti_inv_nll_real={metrics['anti_inv_nll_real']:.4f}")

                if "anti_inv_nll_shuf" in metrics:
                    parts.append(f"anti_inv_nll_shuf={metrics['anti_inv_nll_shuf']:.4f}")

                if "anti_inv_nll_rank" in metrics:
                    parts.append(f"anti_inv_nll_rank={metrics['anti_inv_nll_rank']:.4f}")
                
                if "anti_inv_infonce" in metrics:
                    parts.append(f"anti_inv_infonce={metrics['anti_inv_infonce']:.4f}")

                if 'fw_acc' in metrics:
                    parts.append(f"fw_acc={metrics['fw_acc']:.4f}")
                if 'cdr_acc' in metrics:
                    parts.append(f"cdr_acc={metrics['cdr_acc']:.4f}")
                if 'cdr3_acc' in metrics:
                    parts.append(f"cdr3_acc={metrics['cdr3_acc']:.4f}")
                if 'h3_acc' in metrics:
                    parts.append(f"h3_acc={metrics['h3_acc']:.4f}")
                if 'l3_acc' in metrics:
                    parts.append(f"l3_acc={metrics['l3_acc']:.4f}")
                if 'h3_contact_acc' in metrics:
                    parts.append(f"h3_contact_acc={metrics['h3_contact_acc']:.4f}")
                if 'h3_noncontact_acc' in metrics:
                    parts.append(f"h3_noncontact_acc={metrics['h3_noncontact_acc']:.4f}")
                if 'ar_acc' in metrics:
                    parts.append(f"ar_acc={metrics['ar_acc']:.4f}")
                if 'mlm_acc' in metrics:
                    parts.append(f"mlm_acc={metrics['mlm_acc']:.4f}")
                
                logger.info(f"  [{ds_name}] " + ", ".join(parts))
        
            logger.info("  [Overall Training]")
            parts = []
            if "total_loss" in overall_metrics:
                parts.append(f"loss={overall_metrics['total_loss']:.4f}")
            if overall_metrics.get('ar_loss', 0) > 0:
                parts.append(f"ar_loss={overall_metrics['ar_loss']:.4f}")
            if overall_metrics.get('mlm_loss', 0) > 0:
                parts.append(f"mlm_loss={overall_metrics['mlm_loss']:.4f}")
            if overall_metrics.get('geo_loss', 0) > 0:
                parts.append(f"geo_loss={overall_metrics['geo_loss']:.4f}")
            if overall_metrics.get('if_loss', 0) > 0:
                parts.append(f"if_loss={overall_metrics['if_loss']:.4f}")
            if overall_metrics.get('anti_inv_loss', 0) > 0:
                parts.append(
                    f"anti_inv_loss={overall_metrics['anti_inv_loss']:.4f}"
                )
            if overall_metrics.get('anti_inv_kl', 0) > 0:
                parts.append(f"anti_inv_kl={overall_metrics['anti_inv_kl']:.4f}")
            if overall_metrics.get('anti_inv_nll_real', 0) > 0:
                parts.append(f"anti_inv_nll_real={overall_metrics['anti_inv_nll_real']:.4f}")
            if overall_metrics.get('anti_inv_nll_shuf', 0) > 0:
                parts.append(f"anti_inv_nll_shuf={overall_metrics['anti_inv_nll_shuf']:.4f}")
            if overall_metrics.get('anti_inv_nll_rank', 0) > 0:
                parts.append(f"anti_inv_nll_rank={overall_metrics['anti_inv_nll_rank']:.4f}")
            if overall_metrics.get('anti_inv_infonce', 0) > 0:
                parts.append(f"anti_inv_infonce={overall_metrics['anti_inv_infonce']:.4f}")
            
            if 'fw_acc' in overall_metrics:
                parts.append(f"fw_acc={overall_metrics['fw_acc']:.4f}")
            if 'cdr_acc' in overall_metrics:
                parts.append(f"cdr_acc={overall_metrics['cdr_acc']:.4f}")
            if 'cdr3_acc' in overall_metrics:
                parts.append(f"cdr3_acc={overall_metrics['cdr3_acc']:.4f}")
            if 'h3_acc' in overall_metrics:
                parts.append(f"h3_acc={overall_metrics['h3_acc']:.4f}")
            if 'l3_acc' in overall_metrics:
                parts.append(f"l3_acc={overall_metrics['l3_acc']:.4f}")
            if 'h3_contact_acc' in overall_metrics:
                parts.append(f"h3_contact_acc={overall_metrics['h3_contact_acc']:.4f}")
            if 'h3_noncontact_acc' in overall_metrics:
                parts.append(f"h3_noncontact_acc={overall_metrics['h3_noncontact_acc']:.4f}")
            if 'ar_acc' in overall_metrics:
                parts.append(f"ar_acc={overall_metrics['ar_acc']:.4f}")
            if 'mlm_acc' in overall_metrics:
                parts.append(f"mlm_acc={overall_metrics['mlm_acc']:.4f}")
            
            logger.info("    " + ", ".join(parts))
    
    def _log_validation_metrics(self, val_metrics):
        """Log validation metrics."""
        if not self.is_main:
            return
            
        logger.info("\nValidation Metrics:")
        
        metric_order = [
            'cdr_acc', 'cdr_ppl', 'heavy_cdr3_acc', 'light_cdr3_acc',
            'h3_ppl', 'l3_ppl', 'chain_cdr3_divergence',
            'cdr3_acc', 'cdr3_ppl',
            'cdr3_length_adherence_proxy', 'cdr3_mean_per_sample_acc',
            'ar_acc', 'ar_loss', 'ar_ppl', 'ar_nll',
            'fw_acc', 'mlm_acc', 'mlm_loss', 'mlm_nll',
            'geo_rmsd', 'geo_loss', 'geo_rmsd_unaligned', 'geo_alignment_benefit',
            'geo_ca_ca_mean', 'geo_ca_ca_std', 'geo_ca_ca_deviation',
            'iface_roc_auc', 'iface_pr_auc',
            'residue_iface_roc_auc', 'residue_iface_pr_auc',
            'residue_iface_f1', 'residue_iface_precision', 'residue_iface_recall',
        ]
        
        for dataset_name, metrics in val_metrics.items():
            parts = []
            
            for key in metric_order:
                if key in metrics and metrics[key] is not None:
                    val = metrics[key]
                    if isinstance(val, float) and val == val:
                        parts.append(f"{key}={val:.4f}")
            
            if parts:
                logger.info(f"  [{dataset_name}] " + ", ".join(parts))
            else:
                logger.info(f"  [{dataset_name}] No metrics computed")

        # Overall average metrics
        if val_metrics:
            logger.info("  [Overall Validation]")
            
            all_keys = set()
            for metrics in val_metrics.values():
                all_keys.update(metrics.keys())
            
            important_keys = [k for k in metric_order if k in all_keys]
            
            overall_parts = []
            for key in important_keys:
                vals = [
                    m[key] for m in val_metrics.values() 
                    if key in m and m[key] is not None and 
                    isinstance(m[key], float) and m[key] == m[key]
                ]
                if vals:
                    avg = sum(vals) / len(vals)
                    overall_parts.append(f"{key}={avg:.4f}")
            
            if overall_parts:
                logger.info("    " + ", ".join(overall_parts))

    def _to_device(self, batch):
        """Move batch tensors to device, including nested dicts of tensors."""
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(self.device)
        return batch

    def _prepare_model_inputs(self, batch):
        """Prepare inputs for model forward pass."""
        return {
            "seq_tokens": batch.get("seq_tokens"),
            "seq_tokens_orig": batch.get("seq_tokens_orig"),
            "geom_seq_tokens": batch.get("geom_seq_tokens", None),
            "seq_mask": batch.get("seq_mask"),
            "cdr_mask": batch.get("cdr_mask"),
            "mlm_labels": batch.get("mlm_labels"),
            "coords": batch.get("coords"),
            "coord_mask": batch.get("coord_mask"),
            "antigen_coords": batch.get("antigen_coords"),
            "antigen_coord_mask": batch.get("antigen_coord_mask"),
            "interface_labels": batch.get("interface_labels"),
            "interface_mask": batch.get("interface_mask"),
            "task_mask": batch.get("task_mask"),
            "metadata": batch.get("metadata"),
            "metadata_indices": batch.get("metadata_indices"),
            "chain_mask": batch.get("chain_position_mask"),
            "cdr3_lengths": batch.get("cdr3_lengths"),
            "cdr3_mask": batch.get("cdr3_mask"),
            "antigen_seq_tokens": batch.get("antigen_seq_tokens"),
        }