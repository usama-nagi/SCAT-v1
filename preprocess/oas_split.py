#!/usr/bin/env python
"""
run_once_split.py
=================
One-time post-processing: splits an existing OAS mmap into train/ and val/
subdirectories using a clonotype-disjoint split.

Run:
    nohup python run_once_split.py \
        --mmap_dir /srv/nfs-data/janeway/scat_data/processed/oas_mmap_400M_v2 \
        --workers  7 \
        --seed     42 \
        --val_ratio_hl     0.005 \
        --val_ratio_paired 0.005 \
        > split.log 2>&1 &

    tail -f split.log

The original mmap files are NEVER modified — train/ and val/ are new directories.
Re-running is safe: existing split dirs will be overwritten cleanly.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file copy worker — one subprocess per bin file, no GIL, full CPU
# ─────────────────────────────────────────────────────────────────────────────
def _split_one_file(args):
    """
    Copies rows `combined_idx` from src mmap into a new dst mmap.
    Entirely self-contained — safe to pickle for multiprocessing.
    """
    src_path, dst_path, dtype_str, shape, combined_idx, chunk_size = args

    # dtype comes in as string because numpy dtypes aren't always picklable
    dtype = np.dtype(dtype_str)
    n     = len(combined_idx)

    src = np.memmap(src_path, dtype=dtype, mode="r", shape=shape)

    if src.ndim == 1:
        dst = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(n,))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dst[start:end] = src[combined_idx[start:end]]
    else:
        dst = np.memmap(dst_path, dtype=dtype, mode="w+",
                        shape=(n, shape[1]))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dst[start:end] = src[combined_idx[start:end]]

    dst.flush()
    del dst, src
    return os.path.basename(dst_path)


# ─────────────────────────────────────────────────────────────────────────────
# Clonotype-disjoint split index computation
# ─────────────────────────────────────────────────────────────────────────────
def _disjoint_val_indices(hashes_full, mask, n_val, rng):
    """
    Returns (train_global_idx, val_global_idx) as sorted int32 arrays.

    hashes_full : full clono_hash array (all samples, not just subset)
    mask        : boolean mask selecting the subset (hl or paired)
    n_val       : number of val samples to pull
    rng         : numpy Generator
    """
    global_idx    = np.flatnonzero(mask).astype(np.int32)
    subset_hashes = np.array(hashes_full[global_idx], dtype=np.uint32)

    unique_h, inverse = np.unique(subset_hashes, return_inverse=True)
    sort_order    = np.argsort(inverse, kind="stable")
    sorted_inv    = inverse[sort_order]
    group_starts  = np.searchsorted(sorted_inv, np.arange(len(unique_h)))
    group_ends    = np.append(group_starts[1:], len(sort_order))
    group_perm    = rng.permutation(len(unique_h))

    val_local = []
    for g in group_perm:
        val_local.extend(sort_order[group_starts[g]:group_ends[g]].tolist())
        if len(val_local) >= n_val:
            break

    val_local    = sorted(val_local)
    val_global   = global_idx[val_local]

    train_mask_a = np.ones(len(global_idx), dtype=bool)
    train_mask_a[val_local] = False
    train_global = global_idx[train_mask_a]

    return train_global, val_global


# ─────────────────────────────────────────────────────────────────────────────
# Main split function
# ─────────────────────────────────────────────────────────────────────────────
def split_mmap_to_train_val(mmap_dir, seed, val_ratio_hl,
                             val_ratio_paired, workers):
    """
    Reads the root mmap dir, computes clonotype-disjoint train/val split,
    and writes train/ and val/ subdirectories in parallel.
    The original files are never touched.
    """
    # ── Load metadata ────────────────────────────────────────────────────────
    meta_path = os.path.join(mmap_dir, "meta.json")
    if not os.path.exists(meta_path):
        log(f"ERROR: meta.json not found in {mmap_dir}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    total   = meta["total_samples"]
    max_len = meta["max_len"]
    log(f"mmap_dir  : {mmap_dir}")
    log(f"total     : {total:,} samples")
    log(f"max_len   : {max_len}")

    # ── Verify required files exist ──────────────────────────────────────────
    required = ["clono_hash.bin", "sources.bin"]
    for f in required:
        if not os.path.exists(os.path.join(mmap_dir, f)):
            log(f"ERROR: required file missing: {f}")
            log("       Regenerate mmap with 2_convert_oas_mmap.py (STEP-3 version)")
            sys.exit(1)

    # ── Load sources + hashes (sequential NFS reads — fast) ─────────────────
    log("Loading sources.bin and clono_hash.bin...")
    t0      = time.time()
    sources = np.array(                              # load fully into RAM
        np.memmap(os.path.join(mmap_dir, "sources.bin"),
                  dtype=np.uint8, mode="r", shape=(total,))
    )
    hashes  = np.array(
        np.memmap(os.path.join(mmap_dir, "clono_hash.bin"),
                  dtype=np.uint32, mode="r", shape=(total,))
    )
    log(f"  Loaded in {time.time()-t0:.1f}s  "
        f"(sources {sources.nbytes/1e9:.2f} GB, "
        f"hashes {hashes.nbytes/1e9:.2f} GB)")

    # ── Masks ────────────────────────────────────────────────────────────────
    mask_hl     = (sources != 2)
    mask_paired = (sources == 2)
    n_hl        = int(mask_hl.sum())
    n_paired    = int(mask_paired.sum())

    val_size_hl     = max(1, int(n_hl     * val_ratio_hl))
    val_size_paired = max(1, int(n_paired * val_ratio_paired))

    log(f"hl     : {n_hl:,} total  ->  val={val_size_hl:,} "
        f"({val_ratio_hl*100:.2f}%)")
    log(f"paired : {n_paired:,} total  ->  val={val_size_paired:,} "
        f"({val_ratio_paired*100:.2f}%)")

    # ── Compute disjoint splits ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    log("Computing hl clonotype-disjoint split...")
    t0 = time.time()
    hl_train_idx, hl_val_idx = _disjoint_val_indices(
        hashes, mask_hl, val_size_hl, rng)
    log(f"  hl:     train={len(hl_train_idx):,}  val={len(hl_val_idx):,}  "
        f"({time.time()-t0:.1f}s)")

    log("Computing paired clonotype-disjoint split...")
    t0 = time.time()
    paired_train_idx, paired_val_idx = _disjoint_val_indices(
        hashes, mask_paired, val_size_paired, rng)
    log(f"  paired: train={len(paired_train_idx):,}  val={len(paired_val_idx):,}  "
        f"({time.time()-t0:.1f}s)")

    # Free RAM — no longer needed
    del hashes, sources, mask_hl, mask_paired

    # ── Build file spec list ─────────────────────────────────────────────────
    # Core files — always present in STEP-3 mmaps
    # dtype stored as string so it's picklable across subprocesses
    file_specs = [
        ("tokens.bin",        np.int16,  (total, max_len)),
        ("lengths.bin",       np.int16,  (total,)),
        ("cdr_full_mask.bin", np.uint8,  (total, max_len)),
        ("sources.bin",       np.uint8,  (total,)),
        ("clono_hash.bin",    np.uint32, (total,)),
        ("cdr3_len.bin",      np.uint8,  (total,)),
        ("heavy_len.bin",     np.uint16, (total,)),
    ]

    # Optional individual CDR masks
    for cdr_file in ("cdr1_mask.bin", "cdr2_mask.bin", "cdr3_mask.bin"):
        fp = os.path.join(mmap_dir, cdr_file)
        if os.path.exists(fp):
            file_specs.append((cdr_file, "uint8", (total, max_len)))
            log(f"  Found optional {cdr_file} — will split")
        else:
            log(f"  {cdr_file} absent — skipping (expected for your format)")

    # Legacy CDR3-only mask from old mmap format
    if os.path.exists(os.path.join(mmap_dir, "cdr_mask.bin")):
        file_specs.append(("cdr_mask.bin", "uint8", (total, max_len)))
        log("  Found legacy cdr_mask.bin — will split")

    # Verify all core files exist before starting expensive copies
    missing = [
        fname for fname, _, _ in file_specs
        if not os.path.exists(os.path.join(mmap_dir, fname))
    ]
    if missing:
        log(f"ERROR: missing files: {missing}")
        sys.exit(1)

    # ── Write train and val splits ───────────────────────────────────────────
    for split_name, idx_hl, idx_paired in [
        ("train", hl_train_idx, paired_train_idx),
        ("val",   hl_val_idx,   paired_val_idx),
    ]:
        # Sort combined indices — sequential NFS reads are much faster
        combined_idx = np.sort(
            np.concatenate([idx_hl, idx_paired])
        ).astype(np.int32)
        combined_idx = np.ascontiguousarray(combined_idx)  # picklable

        split_dir = os.path.join(mmap_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        n = len(combined_idx)

        log(f"\n{'─'*60}")
        log(f"Writing {split_name}/  ({n:,} samples, {workers} workers)")
        log(f"{'─'*60}")

        # One task per file — workers copy files independently, no locking
        # Chunk sizes tuned for NFS:
        #   2D files: 5000 rows × 512 cols × dtype_bytes per chunk (~5-10 MB)
        #   1D files: 500K rows per chunk
        tasks = []
        for fname, dtype_str, shape in file_specs:
            src_path = os.path.join(mmap_dir, fname)
            dst_path = os.path.join(split_dir, fname)
            chunk    = 5_000 if len(shape) == 2 else 500_000
            tasks.append((src_path, dst_path, dtype_str,
                          shape, combined_idx, chunk))

        t0 = time.time()
        with Pool(min(workers, len(tasks))) as pool:
            for fname_done in tqdm(
                pool.imap_unordered(_split_one_file, tasks),
                total=len(tasks),
                desc=f"  {split_name}",
            ):
                log(f"    {fname_done} ✓")

        elapsed = time.time() - t0
        log(f"  {split_name}/ written in {elapsed/60:.1f} min")

        # Per-split meta.json
        split_meta = dict(meta)
        split_meta["total_samples"]    = n
        split_meta["source_counts"]    = {
            "hl":     int(len(idx_hl)),
            "paired": int(len(idx_paired)),
        }
        split_meta["split"]            = split_name
        split_meta["split_seed"]       = seed
        split_meta["val_ratio_hl"]     = val_ratio_hl
        split_meta["val_ratio_paired"] = val_ratio_paired
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(split_meta, f, indent=2)
        log(f"  {split_name}/meta.json written")

    log(f"\n{'='*60}")
    log("Pre-split complete.")
    log(f"  Train : {os.path.join(mmap_dir, 'train')}/")
    log(f"  Val   : {os.path.join(mmap_dir, 'val')}/")
    log(f"  Original mmap untouched: {mmap_dir}/")
    log(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="One-time clonotype-disjoint train/val split for OAS mmaps."
    )
    parser.add_argument(
        "--mmap_dir", required=True,
        help="Root mmap directory containing tokens.bin, meta.json, etc."
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Parallel copy workers (default 6). "
             "Each worker handles one bin file. "
             "Keep ≤8 on NFS to avoid saturating the connection."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default 42)"
    )
    parser.add_argument(
        "--val_ratio_hl", type=float, default=0.005,
        help="Fraction of hl samples for validation (default 0.005)"
    )
    parser.add_argument(
        "--val_ratio_paired", type=float, default=0.005,
        help="Fraction of paired samples for validation (default 0.005)"
    )
    args = parser.parse_args()

    log("OAS mmap train/val splitter")
    log(f"  mmap_dir         : {args.mmap_dir}")
    log(f"  workers          : {args.workers}")
    log(f"  seed             : {args.seed}")
    log(f"  val_ratio_hl     : {args.val_ratio_hl}")
    log(f"  val_ratio_paired : {args.val_ratio_paired}")

    split_mmap_to_train_val(
        mmap_dir        = args.mmap_dir,
        seed            = args.seed,
        val_ratio_hl    = args.val_ratio_hl,
        val_ratio_paired= args.val_ratio_paired,
        workers         = args.workers,
    )


if __name__ == "__main__":
    main()