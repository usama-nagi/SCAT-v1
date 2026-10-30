# train/dataset_manager.py
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from data.collate import scat_collate_fn
from data.loaders.oas_loader import OASDataset
from data.loaders.sabdab_loader import SAbDabDataset
from data.loaders.pdb_loader import PDBDataset


DATASET_REGISTRY = {
    "oas": OASDataset,
    "sabdab": SAbDabDataset,
}

class MixedDatasetSampler:
    """
    Samples from multiple datasets based on probabilities.
    Each iteration returns (dataset_name, batch).
    """
    def __init__(self, datasets, probabilities, seed=42, epoch=0, rank=0, world_size=1):
        self.datasets = datasets
        self.seed = seed
        self.epoch = epoch
        self.iterators = None
        self.rank = rank
        self.world_size = world_size
        
        # Single source of truth for probabilities (normalized)
        self.probs = self._normalize(probabilities)
        self.dataset_names = list(self.probs.keys())
        self.prob_values   = [self.probs[n] for n in self.dataset_names]
        
        # RNG for sampling
        self.rng = np.random.RandomState(seed + epoch)
        
        print(f"[MixedDatasetSampler Rank {rank}] Initialized with probabilities:")
        for name, prob in self.probs.items():
            print(f"  {name}: {prob:.3f}")
        self._stale = True
        self._step_counter = 0

    @staticmethod
    def _normalize(probs):
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not hasattr(self, '_stale'):
            self._stale = True
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0

        if self._stale or self.iterators is None:
            if self.iterators is None:
                self.iterators = {}

            # Only build iterators for datasets that are new to this active set.
            # Datasets already in self.iterators are kept alive to preserve their state across probability changes.
            active_set = set(self.dataset_names) & set(self.datasets.keys())
            current_set = set(self.iterators.keys())

            # Tear down iterators for datasets no longer in the active phase
            for name in list(current_set - active_set):
                del self.iterators[name]
                print(f"[MixedDatasetSampler rank={self.rank}] Removed iterator: {name}", flush=True)

            # Create iterators only for genuinely new datasets
            for name in active_set - current_set:
                self.iterators[name] = iter(self.datasets[name])
                print(f"[MixedDatasetSampler rank={self.rank}] Created iterator: {name}", flush=True)

            self._stale = False
            self._step_rng = np.random.RandomState(
                self.seed + self.epoch * 1_000_000
            )
            if self._step_counter > 0:
                for _ in range(self._step_counter):
                    self._step_rng.choice(len(self.dataset_names), p=self.prob_values)

            print(f"[MixedDatasetSampler rank={self.rank}] Active iterators: {list(self.iterators.keys())}", flush=True)

        dataset_idx  = self._step_rng.choice(len(self.dataset_names), p=self.prob_values)
        dataset_name = self.dataset_names[dataset_idx]
        self._step_counter += 1
        return dataset_name, self._next_batch(dataset_name)

    def _next_batch(self, dataset_name: str):
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                return next(self.iterators[dataset_name]) # type: ignore
            except StopIteration:
                if attempt > 0:
                    print(f"[Rank {self.rank}] Resetting iterator for {dataset_name} (attempt {attempt+1})", flush=True)
                self.iterators[dataset_name] = iter(self.datasets[dataset_name]) # type: ignore
        return None
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + epoch)
        self._stale = True
        self._step_counter = 0

    def set_probabilities(self, new_probs: dict):
        self.probs         = self._normalize(new_probs)
        self.dataset_names = list(self.probs.keys())
        self.prob_values   = [self.probs[n] for n in self.dataset_names]
        self._stale        = True   # rebuilt lazily on next __next__() call
        print("[MixedDatasetSampler] Updated sampling probabilities:", flush=True)
        for name, prob in self.probs.items():
            print(f"  {name}: {prob:.3f}", flush=True)

    def get_state(self) -> dict:
        """
        Return a serializable snapshot of the sampler.
        Save this alongside the model checkpoint so that sampling can be
        resumed exactly, including mid-phase probability distributions.
        """
        return {
            "epoch":        self.epoch,
            "probs":        dict(self.probs),
            "rng_state":    self.rng.get_state(),
        }

    def load_state(self, state: dict):
        self.epoch = state["epoch"]
        self.set_probabilities(state["probs"])
        self.rng.set_state(state["rng_state"])
        self._stale = True

class DatasetManager:
    """
    Handles dataset loading with probability-based sampling.
    - Creates individual dataloaders per dataset
    - Returns a mixed sampler that samples ONE dataset per step
    - Probabilities control dataset exposure (not repeat factors)
    - Much simpler and more stable than multi-dataset aggregation
    """
    
    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.datasets = {}
        self.is_main = (rank == 0)
        self.val_datasets = {}

        time.sleep(rank * 0.2)
        
        self._build_datasets()

    def _load_oas_exclude_indices(self, dcfg) -> set:
        """
        Row indices into OASDataset to exclude (CHIMERA-BENCH overlap,
        exact + near-80% identity), precomputed offline against the actual
        292 test-complex CDR-H3/VH sequences. Returns an empty set (and
        warns loudly) until that offline audit produces the index file --
        this is NOT a substitute for running it.
        """
        import os
        exclude_path = dcfg.get("exclude_indices_path")
        if not exclude_path or not os.path.exists(exclude_path):
            if self.is_main:
                print(f"[DatasetManager] WARNING: no OAS exclusion file "
                      f"({exclude_path}) -- OAS is UNAUDITED against CHIMERA-BENCH")
            return set()
        idx = np.load(exclude_path)
        if self.is_main:
            print(f"[DatasetManager] OAS: {len(idx):,} exclusion indices loaded from {exclude_path}")
        return set(idx.tolist())

    def _build_datasets(self):
        """Build all enabled datasets WITHOUT creating huge index lists."""
        
        val_split_overrides = self.cfg["train"].get("val_split_overrides", {})
        default_val_split   = self.cfg["train"].get("val_split", 0.01)
        
        for name, dcfg in self.cfg["data"].items():
            if not dcfg.get("enabled", False):
                if self.is_main:
                    print(f"[DatasetManager] Skipping {name} (disabled)")
                continue

            try:
                cls = DATASET_REGISTRY[name]
                path = dcfg["path"]

                if self.is_main:
                    print(f"[DatasetManager] Loading {name} from {path}...")

                if name == "oas":
                    # Pre-split: load train and val directly - no index gymnastics
                    train_ds_oas = cls(path, split="train")
                    val_ds   = cls(path, split="val")

                    train_sources = np.array(train_ds_oas._sources)  # load fully into RAM - it's only ~400MB uint8
                    val_sources   = np.array(val_ds._sources)

                    exclude_idx = self._load_oas_exclude_indices(dcfg)
                    keep_mask = np.ones(len(train_sources), dtype=bool)
                    if exclude_idx:
                        keep_mask[np.array(sorted(exclude_idx), dtype=np.int64)] = False
                        if self.is_main:
                            print(f"[DatasetManager] OAS: excluding {(~keep_mask).sum():,} "
                                  f"of {len(train_sources):,} train samples (CHIMERA-BENCH overlap)")

                    mask_hl_tr      = (train_sources != 2) & keep_mask
                    mask_paired_tr  = (train_sources == 2) & keep_mask
                    mask_hl_val     = (val_sources   != 2)
                    mask_paired_val = (val_sources   == 2)

                    # Use SORTED indices - preserves sequential NFS read order
                    # DistributedSampler will shuffle these, but within a contiguous
                    # file that's much cheaper than random seeks into 400M-sample mmap
                    hl_tr_idx     = np.flatnonzero(mask_hl_tr).astype(np.int32)      # already sorted
                    paired_tr_idx = np.flatnonzero(mask_paired_tr).astype(np.int32)
                    hl_val_idx    = np.flatnonzero(mask_hl_val).astype(np.int32)
                    paired_val_idx= np.flatnonzero(mask_paired_val).astype(np.int32)

                    # Subset is lighter than FilteredDataset for sorted indices
                    from torch.utils.data import Subset
                    self.datasets["oas_hl"]         = Subset(train_ds_oas, hl_tr_idx.tolist())
                    self.datasets["oas_paired"]     = Subset(train_ds_oas, paired_tr_idx.tolist())
                    self.val_datasets["oas_hl"]     = Subset(val_ds,   hl_val_idx.tolist())
                    self.val_datasets["oas_paired"] = Subset(val_ds,   paired_val_idx.tolist())

                    if self.is_main:
                        print(f"[DatasetManager] oas_hl     train={len(hl_tr_idx):,}  val={len(hl_val_idx):,}")
                        print(f"[DatasetManager] oas_paired train={len(paired_tr_idx):,}  val={len(paired_val_idx):,}")
                    continue
                
                
                dataset = cls(path)
                
                # Check if dataset is empty
                if len(dataset) == 0:
                    print(f"[DatasetManager] WARNING: {name} is empty, removing")
                    continue

                repeat_factor = dcfg.get("repeat_factor", 1)
                val_ratio = val_split_overrides.get(name, default_val_split)
                if name == "sabdab":
                    train_raw, val_raw = self._sabdab_epitope_split(
                        dataset, val_ratio, seed=self.cfg["train"].get("seed", 42))
                else:
                    train_raw, val_raw = self._split_raw(dataset, val_ratio, name)

                # Apply repeat only to train
                if repeat_factor > 1:
                    train_ds = ConcatDataset([train_raw] * repeat_factor)
                    if self.is_main:
                        print(f"[DatasetManager] {name}: "
                            f"raw={len(dataset):,} -> "
                            f"train_raw={len(train_raw):,}x{repeat_factor} "
                            f"val={len(val_raw):,}")
                else:
                    train_ds = train_raw

                self.datasets[name]     = train_ds
                self.val_datasets[name] = val_raw
                if self.is_main:
                    print(f"[DatasetManager] Loaded {name}: {len(dataset):,} samples")

            except Exception as e:
                if self.is_main:
                    print(f"[DatasetManager] Failed to load {name}: {e}")
                    import traceback
                    traceback.print_exc()

        if not self.datasets:
            raise RuntimeError("No datasets loaded! Check paths and enable flags.")

    def _split_raw(self, dataset, val_ratio, name=None):
        """Split a raw (unrepeated) dataset into train/val."""
        total = len(dataset)
        
        # Large OAS-like datasets
        if total > 5_000_000:
            val_size = self.cfg["train"].get("val_size_large", 20000)
            val_size = min(val_size, total - 1)
            seed = self.cfg["train"].get("seed", 42)
            rng  = np.random.default_rng(seed)
            val_idx = np.sort(rng.choice(total, size=val_size, replace=False))
            train_ds = dataset
            val_ds   = Subset(dataset, val_idx.tolist())
            return train_ds, val_ds

        # Small/medium datasets - clean split
        split_point = max(1, min(int(total * (1 - val_ratio)), total - 1))
        indices = np.arange(total)
        rng = np.random.default_rng(self.cfg["train"].get("seed", 42))
        rng.shuffle(indices)
        return (Subset(dataset, indices[:split_point].tolist()),
                Subset(dataset, indices[split_point:].tolist()))
    
    def _sabdab_epitope_split(self, dataset, val_ratio=0.10, seed=42):
        """
        Antigen-disjoint split of SAbDab using epitope_keys cached by
        SAbDabDataset.__init__ (zero extra I/O - keys already loaded).

        Grouping key priority (first non-empty wins, set in sabdab_loader.py):
          1. antigen_cluster_id  - pre-computed MMseqs2/CD-HIT cluster label
          2. target              - free-text SAbDab target name (normalized)
          3. pdb_id[:4]          - last-resort: keeps same crystal out of val
          4. __unk_{i}           - degenerate fallback (same as random split)

        This is the correct split for evaluating antigen-conditioned design:
        val contains antigens never seen during training.
        """
        from collections import defaultdict

        try:
            epitope_keys = getattr(dataset, "epitope_keys", None)
            # Subset wraps the dataset - unwrap to get the underlying keys
            if epitope_keys is None and hasattr(dataset, "dataset"):
                epitope_keys = getattr(dataset.dataset, "epitope_keys", None)
            if not epitope_keys:
                raise ValueError("dataset has no epitope_keys - update sabdab_loader.py")

            n = len(epitope_keys)
            clusters = defaultdict(list)
            for i, key in enumerate(epitope_keys):
                clusters[key].append(i)

            cluster_list = list(clusters.values())
            rng = np.random.default_rng(seed)
            rng.shuffle(cluster_list)

            val_target = max(1, int(n * val_ratio))
            val_idx, train_idx, val_count = [], [], 0
            for cluster in cluster_list:
                if val_count < val_target:
                    val_idx.extend(cluster)
                    val_count += len(cluster)
                else:
                    train_idx.extend(cluster)

            if not train_idx or not val_idx:
                raise ValueError("epitope split produced empty train or val")

            if self.is_main:
                unique_epitopes = len(clusters)
                val_epitopes   = len({epitope_keys[i] for i in val_idx})
                train_epitopes = len({epitope_keys[i] for i in train_idx})
                print(
                    f"[DatasetManager] sabdab EPITOPE split: "
                    f"{unique_epitopes} epitope groups -> "
                    f"train={len(train_idx)} ({train_epitopes} epitopes) "
                    f"val={len(val_idx)} ({val_epitopes} epitopes)"
                )

            return Subset(dataset, train_idx), Subset(dataset, val_idx)

        except Exception as e:
            if self.is_main:
                print(f"[DatasetManager] sabdab epitope split FAILED ({e}), "
                      f"falling back to random split")
            return self._split_raw(dataset, val_ratio, name="sabdab")
    
    @staticmethod
    def _normalize_target(target: str | None) -> str:
        """
        Canonicalize free-text SAbDab target strings so that
        'SARS-CoV-2 spike', 'SARS-CoV-2 Spike protein', 'sars-cov-2 spike'
        all map to the same cluster key.
        """
        import re
        if not target:
            return ""
        t = target.lower().strip()
        t = re.sub(r"[^a-z0-9]+", "_", t)
        t = re.sub(r"_+", "_", t).strip("_")
        return t
    
    def get_loaders(self, epoch=0):
        """
        Create dataloaders and mixed sampler.
        
        Returns:
            mixed_sampler: MixedDatasetSampler for training
            val_loaders: Dict of validation loaders per dataset
        """
        self.train_loaders = {}
        val_loaders = {}
    
        bs = self.cfg["train"]["batch_size"]
        nw = self.cfg["train"].get("num_workers", 4)
        persistent = self.cfg["train"].get("persistent_workers", False)
        prefetch = self.cfg["train"].get("prefetch_factor", 2)
        pin_mem = self.cfg["train"].get("pin_memory", True)
        
        # Get sampling probabilities from config
        sampling_probs = self.cfg["train"]["sampling_probabilities"].get("phase_3", {})

        if not sampling_probs and self.is_main:
            print("[DatasetManager] WARNING: No sampling_probabilities in config!")
            print("  Using uniform sampling (1.0 for all datasets)")
            sampling_probs = {name: 1.0 for name in self.datasets.keys()}
    
        for name, train_ds in self.datasets.items():
            if len(train_ds) == 0:
                continue

            # Create DataLoader for this dataset
            if self.world_size > 1:
                # Distributed training
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=self.cfg["train"]["seed"],
                )
                train_sampler.set_epoch(epoch)
                shuffle = False
            else:
                # Single GPU
                train_sampler = None
                shuffle = True
            
            # Create DataLoader
            effective_len = len(train_sampler) if train_sampler else len(train_ds)

            self.train_loaders[name] = DataLoader(
                train_ds,
                batch_size=bs,
                shuffle=shuffle,
                sampler=train_sampler,
                num_workers=nw,
                collate_fn=scat_collate_fn,
                drop_last=(effective_len >= bs),
                persistent_workers=persistent and (nw > 0),
                prefetch_factor=prefetch if nw > 0 else None,
                pin_memory=pin_mem,
            )
            
            # Validation loader
            val_ds = self.val_datasets.get(name)
            if val_ds is not None and len(val_ds) > 0:
                val_loaders[name] = DataLoader(
                    val_ds,
                    batch_size=bs,
                    shuffle=False,
                    sampler=torch.utils.data.RandomSampler(
                        val_ds,
                        replacement=False,
                        generator=torch.Generator().manual_seed(self.cfg["train"]["seed"])
                    ) if name in ("oas_hl", "oas_paired") else None,
                    num_workers=nw,
                    collate_fn=scat_collate_fn,
                    persistent_workers=persistent and (nw > 0),
                    prefetch_factor=prefetch if nw > 0 else None,
                    pin_memory=pin_mem,
                )
            
            if self.is_main:
                vb = len(val_loaders.get(name, [])) if name in val_loaders else 0
                print(f"[DatasetManager] {name}: "
                    f"train={len(self.train_loaders[name])} batches "
                    f"val={vb} batches")

        if not self.train_loaders:
            raise RuntimeError("No train loaders created!")
        
        all_phase_datasets = set()
        for p in self.cfg["train"]["sampling_probabilities"].values():
            all_phase_datasets.update(p.keys())

        unloaded = all_phase_datasets - set(self.train_loaders.keys())
        if unloaded and self.is_main:
            print(f"[DatasetManager] WARNING: datasets in phase probs but not loaded: {unloaded}")

        unreferenced = set(self.train_loaders.keys()) - all_phase_datasets
        if unreferenced:
            raise ValueError(
                f"Loaded datasets have no sampling probability in any phase: {unreferenced}"
            )

        # Create mixed sampler with probabilities
        mixed_sampler = MixedDatasetSampler(
            datasets=self.train_loaders,
            probabilities=sampling_probs,
            seed=self.cfg["train"]["seed"],
            epoch=epoch,
            rank=self.rank,
            world_size=self.world_size
        )

        return mixed_sampler, val_loaders
    
    def set_epoch(self, epoch):
        for loader in self.train_loaders.values():
            sampler = loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)