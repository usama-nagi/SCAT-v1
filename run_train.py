#!/usr/bin/env python
# ./run_train.py
import os
import yaml
import torch
import re
import warnings
import wandb
import math
from train.scheduler import WarmupCosineScheduler
from model.scat_model import SCATModel
from model.antigen_ablation import AntigenAblation
from model.training_losses import SCATLoss
from model.latent_probe import LatentProbe
from train.trainer import SCATTrainer
from train.dataset_manager import DatasetManager
from utils.seed import set_seed
from utils.logging import get_logger
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_FR_BUFFER_SIZE"] = "1048576"  # 1MB
os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"

def setup_distributed():
    import torch.distributed as dist
    from datetime import timedelta

    # torchrun : RANK, LOCAL_RANK, WORLD_SIZE
    rank       = int(os.environ.get('RANK'))
    local_rank = int(os.environ.get('LOCAL_RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    if world_size == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(minutes=20),
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    props = torch.cuda.get_device_properties(local_rank)
    logger.info(
        f"[Rank {rank}/{world_size}] "
        f"node={os.environ.get('SLURMD_NODENAME', '?')} "
        f"local_rank={local_rank} -> {props.name}"
    )

    return rank, local_rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def save_checkpoint(path, epoch, step, model, optimizer, scheduler, best_val_loss, val_metrics, cfg, sampler=None, scaler=None):
    """Helper to save checkpoint (extracts model from DDP if needed)."""
    model_to_save = model.module if hasattr(model, 'module') else model
    
    ckpt = {
        "epoch": epoch,
        "global_step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
        "config": cfg,
    }
    if sampler is not None:
        ckpt["sampler_state"] = sampler.get_state()
        
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    torch.save(ckpt, path)

# Modules that torch.compile wraps with ._orig_mod. — must match
# exactly what is compiled in the compile block above.
_COMPILED_MODULE_NAMES = [
    "seq_encoder",
    "decoder",
    "cond_proj",
    "lm_head",
    "geom_fusion_proj",
    "seq_adapter",
    "metadata_encoder",
]

def _adapt_checkpoint_state(ckpt_state, model):
    """
    Make checkpoint state_dict compatible with the current model's key naming,
    bridging torch.compile's '_orig_mod.' prefix in either direction.
    Operates per-module prefix so partial compilation is handled correctly:
    e.g. seq_encoder compiled but geom_encoder not — keys are fixed independently.
    Safe no-op when keys already match.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(ckpt_state.keys())

    if not (ckpt_keys ^ model_keys):
        return ckpt_state  # already compatible

    model_has_orig = any("._orig_mod." in k for k in model_keys)
    ckpt_has_orig  = any("._orig_mod." in k for k in ckpt_keys)

    if not model_has_orig and not ckpt_has_orig:
        return ckpt_state  # neither side compiled, nothing to do

    new_state = {}
    n_fixed = 0

    for k, v in ckpt_state.items():
        new_k = k
        for mod_name in _COMPILED_MODULE_NAMES:
            orig_prefix  = f"{mod_name}._orig_mod."   # compiled key form
            plain_prefix = f"{mod_name}."              # uncompiled key form

            if ckpt_has_orig and not model_has_orig:
                # compiled ckpt -> uncompiled model: strip ._orig_mod.
                if k.startswith(orig_prefix):
                    new_k = plain_prefix + k[len(orig_prefix):]
                    n_fixed += 1
                    break

            elif model_has_orig and not ckpt_has_orig:
                # uncompiled ckpt -> compiled model: insert ._orig_mod.
                # Only insert if the model actually compiled this module
                model_has_this_orig = any(
                    mk.startswith(orig_prefix) for mk in model_keys
                )
                if model_has_this_orig and k.startswith(plain_prefix):
                    new_k = orig_prefix + k[len(plain_prefix):]
                    n_fixed += 1
                    break

        new_state[new_k] = v

    if n_fixed > 0:
        direction = "stripped" if (ckpt_has_orig and not model_has_orig) else "inserted"
        print(f"[Resume] torch.compile prefix {direction} for {n_fixed} keys", flush=True)

    return new_state

def try_resume(ckpt_dir, model, optimizer=None, scheduler=None, scaler=None, sampler=None, device="cpu"):
    """Try to resume from checkpoint."""
    latest_path = os.path.join(ckpt_dir, "scat_latest.pt")
    
    # Check for step checkpoints
    step_pattern = re.compile(r"scat_step_(\d+)\.pt")
    best_step = -1
    best_path = None

    for f in os.listdir(ckpt_dir):
        m = step_pattern.match(f)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(ckpt_dir, f)

    if best_path is not None:
        latest_path = best_path
    
    if os.path.exists(latest_path):
        logger.info(f"Resuming from checkpoint: {latest_path}")
        print(f"Resuming from checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        
        model_to_load = model.module if hasattr(model, 'module') else model
        adapted_state = _adapt_checkpoint_state(ckpt["model"], model_to_load)
        missing_keys, unexpected_keys = model_to_load.load_state_dict(
            adapted_state, strict=False
        )
        if unexpected_keys:
            raise RuntimeError(
                f"Checkpoint has keys not present in current model "
                f"(architecture was narrowed?): {unexpected_keys}"
            )
        if missing_keys:
            print(
                f"[Checkpoint] New parameters not in checkpoint - "
                f"initialized fresh: {missing_keys}"
            )
        
        if optimizer and ckpt.get("optimizer") is not None:
            try:
                saved_groups = ckpt.get("optimizer", {}).get("param_groups", [])
                if len(saved_groups) != len(optimizer.param_groups):
                    print(f"WARNING: optimizer param_group count mismatch "
                        f"({len(saved_groups)} saved vs {len(optimizer.param_groups)} current) "
                        f"- skipping optimizer load, starting momentum from zero")
                else:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[Resume] Optimizer state loaded")
                    logger.info("[Resume] Optimizer state loaded")
                    # print("[Checkpoint] Skipping optimizer state due to architecture change")

            except ValueError as e:
                if "parameter group" in str(e).lower() or "size" in str(e).lower():
                    print(
                        f"[Resume] Optimizer state skipped - model architecture "
                        f"changed since checkpoint (new parameters added). "
                        f"Model weights preserved; optimizer moments start fresh. "
                        f"Detail: {e}"
                    )
                else:
                    raise
        
        if scheduler and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info(f"[Resume] Scheduler state restored")
            print(f"[Resume] Scheduler state restored")
        
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
            logger.info("[Resume] GradScaler state restored")
            print("[Resume] GradScaler state restored")

        if sampler is not None:
            if "sampler_state" in ckpt:
                sampler.load_state(ckpt["sampler_state"])
                ss = ckpt["sampler_state"]
                logger.info(f"[Resume] Restored sampler - epoch={ss['epoch']}, probs={ss['probs']}")
            else:
                logger.warning(
                    "[Resume] No sampler_state in checkpoint - "
                    "sampling distribution follows current phase config."
                )
        
        start_epoch = ckpt["epoch"] + 1
        start_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        
        return start_epoch, start_step, best_val_loss
    
    logger.info("No checkpoint found, starting from scratch")
    return 1, 0, float("inf")

def _flatten_metrics(val_metrics: dict) -> dict:
    """
    Flatten {ds_name: {metric: value}} -> {ds_name_metric: value}.
    Drops NaN / Inf / non-numeric values.
    Handles both Python float/int and numpy scalar types.
    """
    flat = {}
    for ds_name, metrics in val_metrics.items():
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                flat[f"{ds_name}_{k}"] = fv
    return flat

# Priority list for best-model selection.
_BEST_MODEL_METRIC_PRIORITY = [
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

def _best_model_score(val_metrics_flat: dict) -> tuple[str | None, float | None]:
    """
    Return (metric_name, value) of the highest-priority metric that exists
    in val_metrics_flat, or (None, None) if none are present.
    Uses explicit None checks - never skips a valid 0.0 score.
    """
    for key in _BEST_MODEL_METRIC_PRIORITY:
        if key in val_metrics_flat:
            return key, val_metrics_flat[key]
    return None, None

def build_testset_loader(cfg, testset_dir):
    """Read-only loader for the held-out test set.
    agforce_exclude_file="" disables the training-time exclusion filter so these
    entries are INCLUDED here (they stay excluded from real training)."""
    from torch.utils.data import DataLoader
    from data.collate import scat_collate_fn
    from data.loaders.sabdab_loader import SAbDabDataset

    ds = SAbDabDataset(
        root_dir=testset_dir,
        max_seq_len=cfg["model"]["sequence_encoder"].get("max_len", 280),
        require_structure=True,
        require_antigen=True,
        exclude_ids=set(),
        agforce_exclude_file="",
        h3_exclude_file="",
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=scat_collate_fn,
        pin_memory=True,
    )

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize distributed training
    rank, local_rank, world_size, device = setup_distributed()
    is_main_process = (rank == 0)

    print(
    f"[PID {os.getpid()}] SLURM_PROCID={os.environ.get('SLURM_PROCID')} "
    f"RANK={os.environ.get('RANK')} -> resolved rank={rank} "
    f"local_rank={local_rank} world_size={world_size} is_main={is_main_process}",
    flush=True
    )
    
    if not is_main_process:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    set_seed(cfg["train"]["seed"] + rank)
    
    if is_main_process:
        logger.info(f"Used Configs: {cfg_path}")
        logger.info(f"Using device: {device}, World size: {world_size}")

    # Dataset & loaders
    dataset_manager = DatasetManager(cfg, rank=rank, world_size=world_size)
    mixed_sampler, val_loaders = dataset_manager.get_loaders(epoch=0)
    
    testset_loader = None
    testset_dir = cfg.get("data", {}).get("agforce", {}).get("path", "")
    if testset_dir and os.path.isdir(testset_dir) and is_main_process:
        try:
            print("[AGFORCEDataset] Loaded id's(exclusion filter disabled)", flush=True)
            testset_loader = build_testset_loader(cfg, testset_dir)
            print(f"[main] AGFORCE loader: {len(testset_loader.dataset)} entries", flush=True)
        except Exception as e:
            print(f"[main] AGFORCE loader failed: {e} - skipping AGFORCE eval", flush=True)

    if is_main_process:
        logger.info("Mixed sampler and validation loaders created")

    # Model
    model = SCATModel(cfg).to(device)
    
    if is_main_process:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if cfg["train"].get("use_compile", False):
        try:
            _compile_kwargs = dict(mode="default", dynamic=True, fullgraph=False)

            # hot path, every batch
            model.seq_encoder = torch.compile(model.seq_encoder, **_compile_kwargs)
            model.decoder     = torch.compile(model.decoder,     **_compile_kwargs)
            model.cond_proj   = torch.compile(model.cond_proj,   **_compile_kwargs)
            model.lm_head     = torch.compile(model.lm_head,     **_compile_kwargs)

            # called on most batches, safe graph
            model.geom_fusion_proj = torch.compile(model.geom_fusion_proj, **_compile_kwargs)
            if getattr(model, "seq_adapter", None) is not None:
                model.seq_adapter = torch.compile(model.seq_adapter, **_compile_kwargs)
            if getattr(model, "metadata_encoder", None) is not None:
                model.metadata_encoder = torch.compile(model.metadata_encoder, **_compile_kwargs)

            if is_main_process:
                compiled = ["seq_encoder", "decoder", "cond_proj", "lm_head", "geom_fusion_proj"]
                if getattr(model, "seq_adapter", None) is not None:
                    compiled.append("seq_adapter")
                if getattr(model, "metadata_encoder", None) is not None:
                    compiled.append("metadata_encoder")
                logger.info(f"[compile] torch.compile(dynamic=True) applied to: {', '.join(compiled)}")
        except Exception as e:
            if is_main_process:
                logger.warning(f"[compile] Failed, falling back to eager: {e}")

    print(f"[Rank {rank}] compile done", flush=True)
    
    # Disable dynamo compilation for antigen xattn modules (sequence length varies)
    for _attr in ["antigen_cross_attn", "antigen_cross_attn_2", "antigen_cross_attn_4"]:
        _mod = getattr(model, _attr, None)
        if _mod is not None:
            torch._dynamo.disable(_mod)  # type: ignore
            
    # synchronize all ranks and all CUDA streams before DDP init
    if torch.cuda.is_available():
        print(f"[Rank {rank}] before cuda synchronize", flush=True)
        torch.cuda.synchronize()
        print(f"[Rank {rank}] after cuda synchronize", flush=True)

    print(f"[Rank {rank}] before DDP", flush=True)
    # Wrap model in DDP if distributed
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        find_unused = cfg["train"].get("find_unused_parameters", False)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,)
        if is_main_process:
            logger.info(f"Model wrapped in DDP (find_unused_parameters={find_unused})")
    print(f"[Rank {rank}] after DDP", flush=True)

    if is_main_process:
        wandb.init(
            project=cfg["train"].get("wandb_project", "SCAT"),
            name=cfg["train"].get("checkpoint_dir", "scat_training_1"),
            config=cfg,
            dir=cfg["train"].get("wandb_dir", "./wandb"),
            resume="allow",
            mode="offline",
            settings=wandb.Settings(init_timeout=30),
        )
        wandb.watch(model, log=None, log_freq=50000)

    # Optimizer
    decay_params = []
    nodecay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Strip compile prefix for classification purposes only
        n_clean = n.replace("._orig_mod.", ".")
        if (n_clean.endswith(".bias")
                or "layernorm" in n_clean.lower()
                or "layer_norm" in n_clean.lower()
                or n_clean.endswith("_gate")
                or n_clean == "null_geom_token"):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(cfg["train"]["lr"]),
    )

    # Scheduler
    steps_per_epoch = cfg["train"].get("steps_per_epoch", 10000)
    
    if is_main_process:
        logger.info(f"Steps per epoch: {steps_per_epoch:,}")
    
    scheduler = None
    if cfg["train"].get("use_scheduler", False):
        accum_steps = max(1, cfg["train"].get("gradient_accumulation_steps", 1))
        max_steps_cfg = cfg["train"].get("max_steps", None)

        if max_steps_cfg:
            sched_total_steps = int(max_steps_cfg)
        else:
            opt_steps_per_epoch = steps_per_epoch // accum_steps
            sched_total_steps = cfg["train"]["epochs"] * opt_steps_per_epoch

        if is_main_process:
            logger.info(f"Scheduler total optimizer steps: {sched_total_steps:,}")

        scheduler = WarmupCosineScheduler(
            optimizer        = optimizer,
            total_epochs     = 1,
            steps_per_epoch  = sched_total_steps, # total steps as a single "epoch"
            warmup_steps     = cfg["train"].get("warmup_steps"),
            warmup_ratio     = cfg["train"].get("warmup_ratio"),
            min_lr           = float(cfg["train"].get("min_lr", 1e-6)),
        )
        
    # Loss function
    loss_fn = SCATLoss(
        vocab_size=cfg["model"]["vocab_size"],
        pad_idx=0,
        loss_weights=cfg.get("loss_weights", {
            "AR": 1.0,
            "MLM": 1.0,
            "GEO": 1.0,
            "IF": 1.0,
        }),
    )

    # Trainer
    trainer = SCATTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        mixed_sampler=mixed_sampler,
        val_loaders=val_loaders,
        device=device,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
    )
    
    # Latent probe and antigen ablation modules (main-process only)
    latent_probe = None
    if is_main_process:
        latent_probe = LatentProbe(
            model        = model,
            cfg          = cfg,
            snapshot_dir = os.path.join(
                               cfg["train"].get("checkpoint_dir", "scat_checkpoints"),
                               "latent_snapshots",
                           ),
        )
    
    antigen_ablation = None
    if is_main_process:
        antigen_ablation = AntigenAblation(
            model       = model,
            device      = device,
        )

    # Resume from checkpoint if available
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch, start_step, best_val_loss = try_resume(
        ckpt_dir,
        model,
        optimizer,
        scheduler,
        scaler=trainer.scaler,
        sampler=mixed_sampler,
        device=device, # type: ignore
    )
    _m = model.module if hasattr(model, 'module') else model
    if hasattr(_m, 'antigen_xattn_gate') and start_step > 100001:
        with torch.no_grad():
            current = _m.antigen_xattn_gate.item()
            if current <= -2.5:
                _m.antigen_xattn_gate.fill_(0.0)
                print(f"[RESUME] Reset antigen_xattn_gate: {current:.4f} -> 0.0 (was stuck)", flush=True)
                for pg in optimizer.param_groups:
                    for p in pg['params']:
                        if p is _m.antigen_xattn_gate:
                            if p in optimizer.state:
                                optimizer.state[p] = {}
                            break
            else:
                print(f"[RESUME] antigen_xattn_gate preserved at {current:.4f}", flush=True)

    torch.cuda.empty_cache()

    # TRAINING LOOP
    max_steps = cfg["train"].get("max_steps", float('inf'))
    max_epochs = cfg["train"]["epochs"]
    
    val_check_interval = cfg["train"].get("val_check_interval", None)
    save_every_n_steps = cfg["train"].get("save_every_n_steps", 10000)
    
    # Early stopping
    early_stop_patience = cfg["train"].get("early_stopping_patience", None)
    early_stop_min_delta = cfg["train"].get("early_stopping_min_delta", 0.001)
    patience_counter = 0
    
    global_step = start_step
    _phase1_scale_reset_done = (start_step >= cfg["train"].get("phase1_end_step", 75_000))
    
    for epoch in range(start_epoch, max_epochs + 1):
        if global_step >= max_steps:
            if is_main_process:
                logger.info(f"Reached max_steps ({max_steps}), stopping training")
            break
        
        if is_main_process:
            logger.info("=" * 60)
            logger.info(f"Epoch {epoch}/{max_epochs} | Global step: {global_step}")
            logger.info("=" * 60)

        if world_size > 1:
            dataset_manager.set_epoch(epoch)
        
        # Update sampler epoch
        mixed_sampler.set_epoch(epoch)

        t1 = cfg["train"].get("phase1_end_step", 75_000)
        t2 = cfg["train"].get("phase2_end_step", 350_000)

        if global_step < t1:
            active_probs = cfg["train"]["sampling_probabilities"]["phase_1"]
            active_phase = 1
        elif global_step < t2:
            active_probs = cfg["train"]["sampling_probabilities"]["phase_2"]
            active_phase = 2
        else:
            active_probs = cfg["train"]["sampling_probabilities"]["phase_3"]
            active_phase = 3

        mixed_sampler.set_probabilities(active_probs)
        trainer.set_active_datasets(list(active_probs.keys()))

        if is_main_process:
            logger.info(f"Updated sampler epoch to {epoch}")
            logger.info(f"Phase {active_phase} active  (t1={t1:,}  t2={t2:,}  step={global_step:,})")
            wandb.log({"sampling/phase": active_phase}, step=global_step)

        schedule = cfg.get("loss_weight_schedule", {})
        phase_key = f"phase_{active_phase}"
        if phase_key in schedule:
            for k, v in schedule[phase_key].items():
                trainer.loss_fn.weights[k] = v
            if is_main_process:
                logger.info(f"[LossSchedule] {phase_key}: {schedule[phase_key]}")
                print(f"[LossSchedule] {phase_key}: {schedule[phase_key]}", flush=True)
        
        # Training with step tracking
        dataset_train_metrics, overall_train_metrics, global_step = trainer.train_epoch(
            epoch, 
            global_step,
            max_steps=max_steps,
            val_check_interval=val_check_interval,
            save_every_n_steps=save_every_n_steps,
            ckpt_dir=ckpt_dir,
            best_val_loss=best_val_loss,
        )
        
        if is_main_process:
            log_dict = {}
        
            # overall metrics
            for k, v in overall_train_metrics.items():
                if isinstance(v, (int, float)):
                    log_dict[f"train/{k}"] = v
        
            # dataset-wise metrics
            for ds_name, metrics in dataset_train_metrics.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"train/{ds_name}/{k}"] = v

            _m = model.module if hasattr(model, 'module') else model
            if hasattr(_m, 'geom_cond_gate') and _m.geom_cond_gate is not None:
                log_dict["gates/geom_gate_beta"] = torch.sigmoid(_m.geom_cond_gate).item()
        
            log_dict["global_step"] = global_step
            log_dict["epoch"] = epoch
            wandb.log(log_dict, step=global_step)

        # End-of-epoch validation and checkpointing
        val_start_epoch = cfg["train"].get("val_start_epoch", 5)
        if val_check_interval is None and epoch >= val_start_epoch:
            val_metrics = trainer.validate(epoch)
            
            if world_size > 1:
                torch.distributed.barrier()
            
            # latent probe runs inside trainer.validate() to ensure it runs on the 
            # same batches as validation metrics, for proper correlation analysis.
            if latent_probe is not None:
                probe_metrics = latent_probe.run(
                    val_loaders = trainer.active_val_loaders if hasattr(trainer, "active_val_loaders")
                                else val_loaders,
                    global_step = global_step,
                    device      = device,
                )

                if is_main_process:
                    wandb.log({
                        f"latent/{k}": v for k, v in probe_metrics.items()
                        if isinstance(v, (int, float))
                    }, step=global_step)

            phase1_end = cfg["train"].get("phase1_end_step", 75_000)

            if not _phase1_scale_reset_done and global_step >= phase1_end:
                print(f"[EarlyStop] Resetting best_val_loss ({best_val_loss:.4f}) to inf - "
                      f"crossed phase1_end_step ({phase1_end}) this epoch; best-metric "
                      f"selection switches from OAS-scale to sabdab_h3_ppl here, prior "
                      f"value is not on a comparable scale", flush=True)
                best_val_loss = float("inf")
                patience_counter = 0
                _phase1_scale_reset_done = True

            if (antigen_ablation is not None
                    and "sabdab" in val_loaders
                    and global_step >= phase1_end):
                ablation_metrics = antigen_ablation.run(
                    sabdab_val_loader = val_loaders["sabdab"],
                    global_step       = global_step,
                )
                if is_main_process:
                    wandb.log({
                        f"ablation/{k}": v
                        for k, v in ablation_metrics.items()
                        if isinstance(v, (int, float))
                    }, step=global_step)
                    print("[AntigenAblation]", "  ".join(
                        f"{k}={v:.4f}" for k, v in ablation_metrics.items()
                    ), flush=True)

            recovery_every_n_epochs = 4
            if (antigen_ablation is not None
                    and "sabdab" in val_loaders
                    and global_step >= phase1_end
                    and (epoch % recovery_every_n_epochs == 0)):
                recovery_metrics = antigen_ablation.cdr3_recovery(
                    sabdab_val_loader = val_loaders["sabdab"],
                    max_batches       = 4,
                )
                if is_main_process:
                    wandb.log({
                        f"recovery/{k}": v
                        for k, v in recovery_metrics.items()
                        if isinstance(v, (int, float))
                    }, step=global_step)
                    print("[AntigenRecovery]", "  ".join(
                        f"{k}={v:.4f}" for k, v in recovery_metrics.items()
                        if isinstance(v, (int, float))
                    ), flush=True)
            
            val_metrics_flat = _flatten_metrics(val_metrics)

            if is_main_process:
                wandb.log(
                    {
                        **{f"val/{k}": v for k, v in val_metrics_flat.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            # Best model selection
            score_key, current_score = _best_model_score(val_metrics_flat)

            if epoch >= val_start_epoch and current_score is not None:
                improvement = best_val_loss - current_score

                if improvement > early_stop_min_delta:
                    best_val_loss = current_score
                    patience_counter = 0

                    if is_main_process:
                        best_path = os.path.join(ckpt_dir, "scat_best.pt")
                        save_checkpoint(
                            best_path, epoch, global_step, model,
                            optimizer, scheduler, best_val_loss,
                            val_metrics, cfg, sampler=mixed_sampler, scaler=trainer.scaler if hasattr(trainer, "scaler") else None,
                        )
                        logger.info(
                            f"New best model | {score_key}={current_score:.4f}"
                        )
                else:
                    patience_counter += 1
                    if is_main_process:
                        logger.info(
                            f"No improvement ({score_key}={current_score:.4f} vs "
                            f"best={best_val_loss:.4f}): "
                            f"{patience_counter}/{early_stop_patience}"
                        )
                
                # Early stopping check
                if early_stop_patience is not None and patience_counter >= early_stop_patience:
                    if is_main_process:
                        logger.info(f"Early stopping after {patience_counter} validations")
                    break
            
            # Save best interface model
            interface_auc = (val_metrics_flat.get("sabdab_residue_iface_roc_auc") or 0.0)
            
            if interface_auc > 0.80 and is_main_process:
                interface_path = os.path.join(ckpt_dir, "scat_interface_best.pt")
                
                should_save = True
                if os.path.exists(interface_path):
                    try:
                        prev = torch.load(interface_path, map_location='cpu', weights_only=False)
                        prev_auc = max(
                            m.get('residue_iface_roc_auc', 0.0) or m.get('iface_roc_auc', 0.0)
                            for m in prev.get('val_metrics', {}).values())
                        should_save = (interface_auc > prev_auc)
                    except:
                        pass
                
                if should_save:
                    save_checkpoint(
                        interface_path, epoch, global_step, model,
                        optimizer, scheduler, best_val_loss,
                        val_metrics, cfg, sampler=mixed_sampler, scaler=trainer.scaler if hasattr(trainer, "scaler") else None,
                    )
                    logger.info(f"Best interface model (AUC={interface_auc:.4f})")
                    
        # Epoch checkpoint
        if is_main_process:
            ckpt_path = os.path.join(ckpt_dir, "scat_latest.pt")
            save_checkpoint(
                ckpt_path, epoch, global_step, model, optimizer,
                scheduler, best_val_loss, {}, cfg, sampler=mixed_sampler, scaler=trainer.scaler if hasattr(trainer, "scaler") else None,
            )
            logger.info(f"Saved epoch checkpoint: {ckpt_path}")

            # Checkpoint cleanup (keep only top-k)
            step_ckpt_pattern = os.path.join(ckpt_dir, "scat_step_*.pt")
            import glob
            step_ckpts = glob.glob(step_ckpt_pattern)

            save_top_k = cfg["train"].get("save_top_k", 3)
            if len(step_ckpts) > save_top_k:
                step_ckpts.sort(key=os.path.getmtime)
                for old_ckpt in step_ckpts[:-save_top_k]:
                    os.remove(old_ckpt)
                    logger.info(f"Removed old checkpoint: {os.path.basename(old_ckpt)}")
        if is_main_process:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        if world_size > 1:
            torch.distributed.barrier()
        
        logger.info("")
        
    # save metric history and detach hooks before exit
    if latent_probe is not None:
        history_path = os.path.join(
            cfg["train"].get("checkpoint_dir", "scat_checkpoints"),
            "latent_history.npz",
        )
        latent_probe.save_history(history_path)
        logger.info(f"Latent history saved to {history_path}")

    # Cleanup
    cleanup_distributed()
    
    if is_main_process:
        wandb.finish()
        logger.info("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/scat_main.yaml",
        help="Path to config YAML file",
    )

    args = parser.parse_args()
    main(args.config)