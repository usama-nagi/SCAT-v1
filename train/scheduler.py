# train/scheduler.py

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Flexible warmup + cosine annealing scheduler.
    
    Supports:
    - Epoch-based warmup (for small datasets)
    - Step-based warmup (for large datasets)
    - Ratio-based warmup (auto-scales with dataset size)
    
    Example usage:
        # Small dataset (epoch-based):
        scheduler = WarmupCosineScheduler(
            optimizer, total_epochs=40, warmup_epochs=1
        )
        
        # Large dataset (step-based):
        scheduler = WarmupCosineScheduler(
            optimizer, total_epochs=8, steps_per_epoch=800000,
            warmup_steps=10000
        )
        
        # Adaptive (ratio-based):
        scheduler = WarmupCosineScheduler(
            optimizer, total_epochs=8, steps_per_epoch=31250000,
            warmup_ratio=0.03  # 3% of training
        )
    """
    
    def __init__(
        self, 
        optimizer, 
        total_epochs,
        steps_per_epoch=None,
        warmup_epochs=None,
        warmup_steps=None,
        warmup_ratio=None,
        min_lr=1e-6,
        last_epoch=-1
    ):
        """
        Args:
            optimizer: Optimizer
            total_epochs: Total training epochs
            steps_per_epoch: Steps per epoch (required if using warmup_steps/ratio)
            warmup_epochs: Warmup duration in epochs
            warmup_steps: Warmup duration in steps (takes precedence)
            warmup_ratio: Warmup as fraction of total training (e.g., 0.03 = 3%)
            min_lr: Minimum learning rate after cosine decay
            last_epoch: Resume from checkpoint
        """
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = float(min_lr)
        
        # Calculate total steps
        if steps_per_epoch is not None:
            self.total_steps = total_epochs * steps_per_epoch
        else:
            self.total_steps = total_epochs  # Fallback to epoch counting
        
        # Determine warmup duration (priority: steps > ratio > epochs)
        if warmup_steps is not None:
            # Fixed step-based warmup
            self.warmup_steps = int(warmup_steps)
            self.warmup_mode = 'steps'
            
        elif warmup_ratio is not None:
            # Ratio-based warmup (scales with dataset)
            self.warmup_steps = int(self.total_steps * warmup_ratio)
            self.warmup_mode = 'steps'
            
        elif warmup_epochs is not None:
            # Epoch-based warmup (legacy, for small datasets)
            if steps_per_epoch is not None:
                self.warmup_steps = int(warmup_epochs * steps_per_epoch)
                self.warmup_mode = 'steps'
            else:
                self.warmup_epochs = int(warmup_epochs)
                self.warmup_mode = 'epochs'
        else:
            # Default: 3% of training
            self.warmup_steps = max(1000, int(self.total_steps * 0.03))
            self.warmup_mode = 'steps'
            print(f"[Scheduler] No warmup specified, using default: "
                  f"{self.warmup_steps:,} steps (3% of training)")
        
        # Logging
        if self.warmup_mode == 'steps':
            warmup_pct = self.warmup_steps / self.total_steps * 100
            print(f"[Scheduler] Warmup: {self.warmup_steps:,} steps "
                  f"({warmup_pct:.1f}% of {self.total_steps:,} total steps)")
            print(f"[Scheduler] Cosine annealing: {self.total_steps - self.warmup_steps:,} steps")
        else:
            print(f"[Scheduler] Warmup: {self.warmup_epochs} epochs (legacy mode)")
            print(f"[Scheduler] Cosine annealing: {self.total_epochs - self.warmup_epochs} epochs")
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self): # type: ignore
        """Get learning rate for current step."""
        
        # Determine current position in training
        if self.warmup_mode == 'steps':
            current_step = self.last_epoch  # last_epoch is actually step count
            warmup_duration = self.warmup_steps
            total_duration = self.total_steps
        else:
            current_step = self.last_epoch
            warmup_duration = self.warmup_epochs
            total_duration = self.total_epochs
        
        # Phase 1: Linear Warmup
        if current_step < warmup_duration:
            alpha = current_step / warmup_duration
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Phase 2: Cosine Annealing
        else:
            progress = (current_step - warmup_duration) / (total_duration - warmup_duration)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            # Cosine decay
            cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine 
                for base_lr in self.base_lrs
            ]


class AdaptiveWarmupScheduler:
    """
    Automatically selects warmup strategy based on dataset size.
    
    Decision rules:
    - Dataset < 1M samples: Epoch-based warmup (1 epoch)
    - Dataset ≥ 1M samples: Step-based warmup (3% of training)
    
    This ensures consistent warmup behavior across scales.
    """
    
    @staticmethod
    def create(
        optimizer,
        total_epochs,
        dataset_size,
        batch_size,
        min_lr=1e-6,
    ):
        """
        Factory method to create appropriate scheduler.
        
        Args:
            optimizer: Optimizer
            total_epochs: Total training epochs
            dataset_size: Total number of training samples
            batch_size: Batch size
            min_lr: Minimum learning rate
        
        Returns:
            WarmupCosineScheduler instance
        """
        steps_per_epoch = dataset_size // batch_size
        total_steps = total_epochs * steps_per_epoch
        
        print(f"\n[AdaptiveScheduler] Dataset analysis:")
        print(f"  Total samples: {dataset_size:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Total steps: {total_steps:,}")
        
        # Decision logic
        if dataset_size < 1_000_000:
            # Small dataset: epoch-based warmup
            warmup_epochs = 1 if total_epochs >= 10 else 0
            
            print(f"\n[AdaptiveScheduler] Small dataset mode:")
            print(f"  Strategy: Epoch-based warmup")
            print(f"  Warmup: {warmup_epochs} epoch(s)")
            
            return WarmupCosineScheduler(
                optimizer=optimizer,
                total_epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=warmup_epochs,
                min_lr=min_lr,
            )
        else:
            # Large dataset: step-based warmup (3% of total)
            warmup_ratio = 0.03
            warmup_steps = int(total_steps * warmup_ratio)
            
            print(f"\n[AdaptiveScheduler] Large dataset mode:")
            print(f"  Strategy: Step-based warmup")
            print(f"  Warmup ratio: {warmup_ratio*100:.1f}%")
            print(f"  Warmup steps: {warmup_steps:,}")
            
            return WarmupCosineScheduler(
                optimizer=optimizer,
                total_epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                warmup_ratio=warmup_ratio,
                min_lr=min_lr,
            )


# ============================================================================
# Helper function for easy integration
# ============================================================================

def create_scheduler(optimizer, cfg, dataset_size=None, steps_per_epoch=None):
    """
    Convenience function to create scheduler from config.
    
    Args:
        optimizer: Optimizer
        cfg: Config dict with 'train' section
        dataset_size: Total training samples (optional, for adaptive mode)
        steps_per_epoch: Steps per epoch (optional, for step-based warmup)
    
    Returns:
        Scheduler instance or None if disabled
    """
    if not cfg["train"].get("use_scheduler", False):
        print("[Scheduler] Disabled")
        return None
    
    total_epochs = cfg["train"]["epochs"]
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    
    # Check for adaptive mode
    if cfg["train"].get("warmup_mode") == "adaptive":
        if dataset_size is None or steps_per_epoch is None:
            raise ValueError("Adaptive mode requires dataset_size and steps_per_epoch")
        
        batch_size = cfg["train"]["batch_size"]
        return AdaptiveWarmupScheduler.create(
            optimizer=optimizer,
            total_epochs=total_epochs,
            dataset_size=dataset_size,
            batch_size=batch_size,
            min_lr=min_lr,
        )
    
    # Manual configuration
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=cfg["train"].get("warmup_epochs"),
        warmup_steps=cfg["train"].get("warmup_steps"),
        warmup_ratio=cfg["train"].get("warmup_ratio"),
        min_lr=min_lr,
    )