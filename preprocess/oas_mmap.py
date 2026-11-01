"""
2_convert_oas_mmap.py
================================================
  - clonotype_hash()        : V-gene + J-gene + CDR3 string -> uint32 key
  - tokenize_sample()       : returns clono_hash + cdr3_len; adds 5 hard filters
  - open_writers()          : adds clono_hash.bin + cdr3_len.bin
  - write_result()          : writes two new bins
  - process_file()          : passes clono_hash/cdr3_len through; trim block updated
  - meta.json               : records new fields

WHY regenerate mmaps (not patch retroactively):
  - V+J+CDR3 clonotype key is ~10x more specific than CDR3-token-hash alone
  - Hard filters only work at source (.pt shard) level where cdr3_aa string exists
  - build_cdr3_hashes.py is no longer needed

Run:
    nohup python 2_convert_oas_mmap.py \\
        --oas_root /processed_oas \\
        --out_dir  /oas_mmap_400M_v2 \\
        --target_total 400_000_000 \\
        --max_len 512 \\
        --workers 16 > convert.log 2>&1 &
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

# Vocab (unchanged) 
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD, UNK, START, END, MASK = "<pad>", "<unk>", "<s>", "</s>", "<mask>"
VALID_AA_SET = set(AMINO_ACIDS)

def build_vocab():
    toks = [PAD, UNK, START, END, MASK] + list(AMINO_ACIDS)
    stoi = {s: i for i, s in enumerate(toks)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

STOI, ITOS = build_vocab()
SEED       = 42
PAD_ID     = STOI[PAD]
UNK_ID     = STOI[UNK]
SOURCE_MAP = {"heavy": 0, "light": 1, "paired": 2}

# Hard-filter constants 
CDR3_MIN_LEN     = 3    # residues - below this it's an annotation error
CDR3_MAX_LEN     = 35   # residues - above this is biologically implausible
MIN_CDR_TOKENS   = 3    # minimum CDR-masked positions in tokenised output
VALID_V_PREFIXES = ("IGHV", "IGLV", "IGKV", "TRAV", "TRBV")  # basic sanity

# CDR masking helper (unchanged) 
def create_cdr_masks(seq, cdr1, cdr2, cdr3, max_len):
    m1 = np.zeros(max_len, dtype=np.uint8)
    m2 = np.zeros(max_len, dtype=np.uint8)
    m3 = np.zeros(max_len, dtype=np.uint8)
    for mask, cdr in ((m1, cdr1), (m2, cdr2), (m3, cdr3)):
        if cdr:
            s = seq.find(cdr)
            if s != -1:
                mask[s : min(s + len(cdr), max_len)] = 1
    full = (m1 | m2 | m3).astype(np.uint8)
    return m1, m2, m3, full


# 
# STEP-3 ADDITION A: Clonotype hash
# 
import hashlib as _hashlib

def clonotype_hash(v_call, j_call, cdr3_aa) -> np.uint32:
    """
    V-gene prefix (7 chars) + J-gene prefix (5 chars) + CDR3 string -> uint32.

    Uses hashlib.sha256 (NOT Python's built-in hash()) so the value is
    identical across processes, Python versions, and operating systems.
    Python's hash() is randomized by PYTHONHASHSEED - using it would make
    clono_hash.bin inconsistent between mmap creation and training time.
    """
    v_prefix = (v_call  or "")[:7]
    j_prefix = (j_call  or "")[:5]
    cdr3_str = (cdr3_aa or "")
    key      = f"{v_prefix}|{j_prefix}|{cdr3_str}".encode("ascii", errors="replace")
    digest   = _hashlib.sha256(key).digest()
    # Take first 4 bytes as uint32 (little-endian)
    return np.frombuffer(digest[:4], dtype=np.uint32)[0]


# 
# STEP-3 ADDITION B: Hard quality filters
# 
def _cdr3_passes_hard_filters(cdr3_aa: str, seq: str, v_call: str) -> tuple:
    """
    Returns (passes: bool, reason: str).
    All five filters must pass for the sequence to enter the mmap.

    Filter 1 - CDR3 length:
      3-35 residues.  Catches annotation errors that slipped through
      upstream validation (e.g. CDR3="A" or CDR3=80-residue artefacts).

    Filter 2 - CDR3 standard amino acids only:
      No X, *, -, B, Z.  These appear in low-quality OAS entries.

    Filter 3 - CDR3 locatable in sequence:
      If cdr3_aa cannot be found as a substring, the annotation is wrong.
      This happens when ANARCI and IMGT disagree on CDR boundaries.

    Filter 4 - V-gene format:
      If v_call is provided it must start with a known immunoglobulin prefix.
      Catches mislabelled or chimeric entries.

    Filter 5 - CDR3 is not pure low-complexity:
      A CDR3 that is entirely one amino acid (e.g. "AAAAAAAAAAAA") is almost
      certainly a sequencing artefact.
    """
    if not cdr3_aa:
        return False, "empty_cdr3"

    # 1. Length
    L = len(cdr3_aa)
    if not (CDR3_MIN_LEN <= L <= CDR3_MAX_LEN):
        return False, f"cdr3_len_{L}"

    # 2. Standard AAs only
    if not set(cdr3_aa).issubset(VALID_AA_SET):
        return False, "cdr3_non_standard_aa"

    # 3. Locatable in full sequence
    if seq and seq.find(cdr3_aa) == -1:
        return False, "cdr3_not_in_seq"

    # 4. V-gene format (only checked when v_call is provided)
    if v_call and not any(v_call.startswith(p) for p in VALID_V_PREFIXES):
        return False, f"bad_v_prefix_{v_call[:6]}"

    # 5. Not pure low-complexity
    if L > 4 and len(set(cdr3_aa)) == 1:
        return False, "cdr3_low_complexity"

    return True, "ok"


# 
# Tokenisation (extended to return clono_hash, cdr3_len)
# 
def tokenize_sample(sample, source, max_len):
    """
    Returns result dict with new keys: clono_hash (uint32), cdr3_len (uint8).
    Returns None if sample fails hard filters.
    """
    if source == "paired":
        h      = sample.get("heavy") or {}
        l      = sample.get("light") or {}
        h_s    = h.get("sequence_aa") or ""
        l_s    = l.get("sequence_aa") or ""
        seq    = h_s + l_s
        # Extract gene calls and CDR3 from heavy chain (primary binding chain)
        v_call   = h.get("v_call") or ""
        j_call   = h.get("j_call") or ""
        cdr3_aa  = h.get("cdr3_aa") or ""

        #  Hard filter on heavy CDR3 
        passes, reason = _cdr3_passes_hard_filters(cdr3_aa, h_s, v_call)
        if not passes:
            return None

        h1, h2, h3, _ = create_cdr_masks(h_s,
            h.get("cdr1_aa", ""), h.get("cdr2_aa", ""), cdr3_aa, len(h_s))
        l1, l2, l3, _ = create_cdr_masks(l_s,
            l.get("cdr1_aa", ""), l.get("cdr2_aa", ""), l.get("cdr3_aa", ""), len(l_s))

        total = len(h_s) + len(l_s)
        m1 = np.zeros(max_len, dtype=np.uint8)
        m2 = np.zeros(max_len, dtype=np.uint8)
        m3 = np.zeros(max_len, dtype=np.uint8)
        m1[:len(h_s)] = h1; m2[:len(h_s)] = h2; m3[:len(h_s)] = h3
        if total <= max_len:
            m1[len(h_s):total] = l1
            m2[len(h_s):total] = l2
            m3[len(h_s):total] = l3
        cdr_full = (m1 | m2 | m3).astype(np.uint8)

    else:
        seq      = sample.get("sequence_aa") or ""
        v_call   = sample.get("v_call") or ""
        j_call   = sample.get("j_call") or ""
        cdr3_aa  = sample.get("cdr3_aa") or ""

        #  Hard filter 
        passes, reason = _cdr3_passes_hard_filters(cdr3_aa, seq, v_call)
        if not passes:
            return None

        m1, m2, m3, cdr_full = create_cdr_masks(
            seq,
            sample.get("cdr1_aa", ""), sample.get("cdr2_aa", ""),
            cdr3_aa, max_len,
        )

    if not seq:
        return None

    # Tokenize
    tok = np.full(max_len, PAD_ID, dtype=np.int16)
    ids = [STOI.get(a, UNK_ID) for a in seq[:max_len]]
    tok[:len(ids)] = ids

    # Filter 6: minimum CDR token coverage in output
    if int(cdr_full[:len(ids)].sum()) < MIN_CDR_TOKENS:
        return None

    #  STEP-3: clonotype hash and CDR3 length 
    c_hash    = clonotype_hash(v_call, j_call, cdr3_aa)
    cdr3_len  = np.uint8(min(len(cdr3_aa), 255))
    # heavy_len: number of heavy-chain tokens in the final (possibly clipped) output.
    # For unpaired sequences this is identical to len(ids); the field is only
    # semantically meaningful for paired sequences but we store it everywhere so
    # every row in heavy_len.bin is valid and no branch is needed at load time.
    if source == "paired":
        h_s        = sample.get("heavy", {}).get("sequence_aa") or ""
        heavy_len  = np.uint16(min(len(h_s), max_len))
    else:
        heavy_len  = np.uint16(min(len(ids), max_len))

    return dict(
        tokens        = tok,
        length        = len(ids),
        cdr3_mask     = m3,
        cdr_full_mask = cdr_full,
        source        = SOURCE_MAP[source],
        clono_hash    = c_hash,
        cdr3_len      = cdr3_len,
        heavy_len     = heavy_len,
    )


# 
# File worker
# 
def process_file(args):
    source, fpath, max_len, keep_count = args
    try:
        data    = torch.load(fpath, map_location="cpu", weights_only=False)
        samples = data if isinstance(data, list) else data.get("samples", [])
    except Exception:
        return None

    if not samples:
        return None

    random.shuffle(samples)
    if keep_count is not None:
        samples = samples[:keep_count]

    results = []
    for s in samples:
        r = tokenize_sample(s, source, max_len)
        if r is not None:
            results.append(r)

    if not results:
        return None

    return dict(
        source     = source,
        tokens     = np.stack([r["tokens"]        for r in results]),
        lengths    = np.array([r["length"]         for r in results], dtype=np.int16),
        cdr3       = np.stack([r["cdr3_mask"]      for r in results]),
        cdr_full   = np.stack([r["cdr_full_mask"]  for r in results]),
        sources    = np.full(len(results), SOURCE_MAP[source], dtype=np.uint8),
        clono_hash = np.array([r["clono_hash"] for r in results], dtype=np.uint32),
        cdr3_len   = np.array([r["cdr3_len"]   for r in results], dtype=np.uint8),
        heavy_len  = np.array([r["heavy_len"]  for r in results], dtype=np.uint16),
        count      = len(results),
    )


# 
# File collection (unchanged)
# 
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_files(oas_root, source):
    src_dir = Path(oas_root) / source
    if not src_dir.is_dir():
        log(f"WARNING: {src_dir} not found, skipping {source}")
        return []
    files = sorted(src_dir.glob("*.pt"))
    log(f"  {source}: {len(files)} .pt files found")
    return files


# 
# Memmap writers  - STEP-3: adds clono_hash.bin + cdr3_len.bin
# 
def open_writers(out_dir):
    return {
        "tokens"    : open(os.path.join(out_dir, "tokens.bin"),         "wb"),
        "lengths"   : open(os.path.join(out_dir, "lengths.bin"),         "wb"),
        "cdr3"      : open(os.path.join(out_dir, "cdr3_mask.bin"),       "wb"),
        "cdr_full"  : open(os.path.join(out_dir, "cdr_full_mask.bin"),   "wb"),
        "sources"   : open(os.path.join(out_dir, "sources.bin"),         "wb"),
        "clono_hash": open(os.path.join(out_dir, "clono_hash.bin"),      "wb"),
        "cdr3_len"  : open(os.path.join(out_dir, "cdr3_len.bin"),        "wb"),
        "heavy_len" : open(os.path.join(out_dir, "heavy_len.bin"),       "wb"),
    }


def write_result(writers, result):
    writers["tokens"    ].write(result["tokens"    ].tobytes())
    writers["lengths"   ].write(result["lengths"   ].tobytes())
    writers["cdr3"      ].write(result["cdr3"      ].tobytes())
    writers["cdr_full"  ].write(result["cdr_full"  ].tobytes())
    writers["sources"   ].write(result["sources"   ].tobytes())
    writers["clono_hash"].write(result["clono_hash"].tobytes())
    writers["cdr3_len"  ].write(result["cdr3_len"  ].tobytes())
    writers["heavy_len" ].write(result["heavy_len" ].tobytes())


def close_writers(writers):
    for f in writers.values():
        f.close()


# 
# run_source  - STEP-3: quota-trim block updated
# 
def run_source(source, files, max_len, quota, keep_per_file, workers, writers,
               source_counts, cdr_stats, filter_stats):
    if quota <= 0 or not files:
        return 0

    tasks   = [(source, str(f), max_len, keep_per_file) for f in files]
    written = 0
    t0      = time.time()

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_file, tasks, chunksize=4),
            total=len(tasks),
            desc=f"  {source}",
        ):
            if result is None:
                continue
            if written >= quota:
                break

            remaining = quota - written
            if result["count"] > remaining:
                for k in ("tokens", "lengths", "cdr3", "cdr_full",
                          "sources", "clono_hash", "cdr3_len", "heavy_len"):
                    result[k] = result[k][:remaining] # type: ignore
                result["count"] = remaining

            write_result(writers, result)
            written              += result["count"]
            source_counts[source]+= result["count"]
            cdr_stats["cdr3"]    += int((result["cdr3"   ].sum(1) > 0).sum()) # type: ignore
            cdr_stats["any_cdr"] += int((result["cdr_full"].sum(1) > 0).sum()) # type: ignore

    elapsed = time.time() - t0
    log(f"  {source}: {written:,} samples in {elapsed/60:.1f} min")
    return written

# 
# Parallel split worker - runs in a subprocess, one per file
# 
def _split_one_file(args):
    """
    Worker function: copies rows `combined_idx` from src file into dst file.
    Runs in a separate process - no GIL, full CPU core usage.
    """
    src_path, dst_path, dtype, shape, combined_idx, chunk_size = args

    n   = len(combined_idx)
    src = np.memmap(src_path, dtype=dtype, mode="r", shape=shape)

    if src.ndim == 1:
        dst = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(n,))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dst[start:end] = src[combined_idx[start:end]]
    else:
        dst = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(n, shape[1]))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dst[start:end] = src[combined_idx[start:end]]

    dst.flush()
    del dst, src
    return os.path.basename(dst_path)


def split_mmap_to_train_val(out_dir, seed=42, val_ratio_hl=0.005,
                             val_ratio_paired=0.005, workers=4):
    """
    One-time post-processing: split the written mmap into train/ and val/ subdirs.
    Uses multiprocessing to copy files in parallel - one process per bin file.

    workers: number of parallel copy processes.
             4 is safe on NFS (avoids saturating the NFS connection).
             Increase to 8 if your NFS has high bandwidth and low latency.
    """
    log("Post-processing: computing clonotype-disjoint train/val split...")

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    total   = meta["total_samples"]
    max_len = meta["max_len"]

    hashes  = np.memmap(
        os.path.join(out_dir, "clono_hash.bin"),
        dtype=np.uint32, mode="r", shape=(total,)
    )
    sources = np.memmap(
        os.path.join(out_dir, "sources.bin"),
        dtype=np.uint8, mode="r", shape=(total,)
    )

    mask_hl     = (sources != 2)
    mask_paired = (sources == 2)

    n_hl     = int(mask_hl.sum())
    n_paired = int(mask_paired.sum())
    val_size_hl     = max(1, int(n_hl     * val_ratio_hl))
    val_size_paired = max(1, int(n_paired * val_ratio_paired))

    log(f"  hl total={n_hl:,}  -> val={val_size_hl:,} ({val_ratio_hl*100:.2f}%)")
    log(f"  paired total={n_paired:,}  -> val={val_size_paired:,} ({val_ratio_paired*100:.2f}%)")

    rng = np.random.default_rng(seed)

    def disjoint_val_indices(mask, n_val):
        global_idx    = np.flatnonzero(mask).astype(np.int32)
        subset_hashes = np.array(hashes[global_idx], dtype=np.uint32)
        unique_h, inverse = np.unique(subset_hashes, return_inverse=True)
        sort_order    = np.argsort(inverse, kind="stable")
        sorted_inv    = inverse[sort_order]
        group_starts  = np.searchsorted(sorted_inv, np.arange(len(unique_h)))
        group_ends    = np.append(group_starts[1:], len(sort_order))
        group_perm    = rng.permutation(len(unique_h))
        val_local     = []
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

    log("Computing hl split...")
    hl_train_idx, hl_val_idx = disjoint_val_indices(mask_hl, val_size_hl)
    log(f"  hl: train={len(hl_train_idx):,}  val={len(hl_val_idx):,}")

    log("Computing paired split...")
    paired_train_idx, paired_val_idx = disjoint_val_indices(mask_paired, val_size_paired)
    log(f"  paired: train={len(paired_train_idx):,}  val={len(paired_val_idx):,}")

    # Build file specs - core files always present
    file_specs = [
        ("tokens.bin",        np.int16,  (total, max_len)),
        ("lengths.bin",       np.int16,  (total,)),
        ("cdr_full_mask.bin", np.uint8,  (total, max_len)),
        ("sources.bin",       np.uint8,  (total,)),
        ("clono_hash.bin",    np.uint32, (total,)),
        ("cdr3_len.bin",      np.uint8,  (total,)),
        ("heavy_len.bin",     np.uint16, (total,)),
    ]
    for cdr_file in ("cdr1_mask.bin", "cdr2_mask.bin", "cdr3_mask.bin"):
        if os.path.exists(os.path.join(out_dir, cdr_file)):
            file_specs.append((cdr_file, np.uint8, (total, max_len)))
            log(f"  Found optional {cdr_file} - will split")
        else:
            log(f"  {cdr_file} not found - skipping (optional)")
    if os.path.exists(os.path.join(out_dir, "cdr_mask.bin")):
        file_specs.append(("cdr_mask.bin", np.uint8, (total, max_len)))
        log("  Found legacy cdr_mask.bin - will split")

    # Process train and val splits
    for split_name, idx_hl, idx_paired in [
        ("train", hl_train_idx, paired_train_idx),
        ("val",   hl_val_idx,   paired_val_idx),
    ]:
        combined_idx = np.sort(np.concatenate([idx_hl, idx_paired]))
        # Convert to plain numpy array - must be picklable for multiprocessing
        combined_idx = np.ascontiguousarray(combined_idx, dtype=np.int32)

        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        n = len(combined_idx)

        log(f"Writing {split_name} ({n:,} samples) to {split_dir}/ "
            f"[{workers} parallel workers]...")

        # Build task list - one task per file
        # Chunk size tuned for NFS: large enough to amortize seek latency,
        # small enough to fit in CPU cache.
        # 2D files (tokens, masks): 5000 rows × 512 cols × 2 bytes = 5MB per chunk
        # 1D files (lengths, sources etc): 500K rows per chunk
        tasks = []
        for fname, dtype, shape in file_specs:
            src_path = os.path.join(out_dir, fname)
            if not os.path.exists(src_path):
                continue
            dst_path  = os.path.join(split_dir, fname)
            chunk     = 5_000 if len(shape) == 2 else 500_000
            tasks.append((src_path, dst_path, dtype, shape, combined_idx, chunk))

        # Run parallel copies
        # Each worker handles one file independently - no shared state,
        # no locks needed. NFS handles concurrent writes to different files fine.
        t0 = time.time()
        with Pool(min(workers, len(tasks))) as pool:
            for fname_done in tqdm(
                pool.imap_unordered(_split_one_file, tasks),
                total=len(tasks),
                desc=f"  {split_name}",
            ):
                log(f"    {fname_done} done")
        log(f"  {split_name} files written in {(time.time()-t0)/60:.1f} min")

        # Write per-split metadata
        split_meta = dict(meta)
        split_meta["total_samples"]      = n
        split_meta["source_counts"]      = {
            "hl":     int(len(idx_hl)),
            "paired": int(len(idx_paired)),
        }
        split_meta["split"]              = split_name
        split_meta["split_seed"]         = seed
        split_meta["val_ratio_hl"]       = val_ratio_hl
        split_meta["val_ratio_paired"]   = val_ratio_paired
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(split_meta, f, indent=2)
        log(f"  {split_name}/meta.json written")

    log("Pre-split complete.")
    log(f"  Train: {out_dir}/train/")
    log(f"  Val:   {out_dir}/val/")

# 
# Main
# 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oas_root",     required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--target_total", type=int, required=True)
    parser.add_argument("--max_len",      type=int, default=512)
    parser.add_argument("--workers",      type=int, default=16)
    parser.add_argument("--seed",         type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"Target total : {args.target_total:,}")
    log(f"Output dir   : {args.out_dir}")
    log(f"Hard filters : CDR3 len [{CDR3_MIN_LEN}-{CDR3_MAX_LEN}], "
        f"min CDR tokens {MIN_CDR_TOKENS}, standard AAs only, V-gene check")

    paired_files = collect_files(args.oas_root, "paired")
    heavy_files  = collect_files(args.oas_root, "heavy")
    light_files  = collect_files(args.oas_root, "light")

    est_paired = len(paired_files) * 50_000
    log(f"\nEstimated paired upper bound : {est_paired:,}")

    paired_quota = min(est_paired, args.target_total)
    n_hl     = len(heavy_files) + len(light_files)
    h_ratio  = len(heavy_files) / n_hl if n_hl > 0 else 0.5

    writers       = open_writers(args.out_dir)
    source_counts = {"heavy": 0, "light": 0, "paired": 0}
    cdr_stats     = {"cdr3": 0, "any_cdr": 0}
    filter_stats  = {}   # reserved for future per-reason counters

    log("\n Phase 1: paired ")
    paired_written = run_source(
        "paired", paired_files, args.max_len,
        quota=args.target_total, keep_per_file=None,
        workers=args.workers, writers=writers,
        source_counts=source_counts, cdr_stats=cdr_stats,
        filter_stats=filter_stats,
    )

    remainder   = args.target_total - paired_written
    heavy_quota = min(int(round(remainder * h_ratio)), len(heavy_files) * 50_000)
    light_quota = remainder - heavy_quota
    h_per_file  = int(heavy_quota / max(len(heavy_files), 1) * 1.1) if heavy_files else 0
    l_per_file  = int(light_quota / max(len(light_files), 1) * 1.1) if light_files else 0

    log(f"\n Phase 2: heavy quota={heavy_quota:,} ~{h_per_file:,}/file ")
    run_source("heavy", heavy_files, args.max_len,
               quota=heavy_quota, keep_per_file=h_per_file,
               workers=args.workers, writers=writers,
               source_counts=source_counts, cdr_stats=cdr_stats,
               filter_stats=filter_stats)

    log(f"\n Phase 3: light quota={light_quota:,} ~{l_per_file:,}/file ")
    run_source("light", light_files, args.max_len,
               quota=light_quota, keep_per_file=l_per_file,
               workers=args.workers, writers=writers,
               source_counts=source_counts, cdr_stats=cdr_stats,
               filter_stats=filter_stats)

    close_writers(writers)
    
    total_written = sum(source_counts.values())
    meta = {
        "total_samples"    : total_written,
        "max_len"          : args.max_len,
        "vocab_size"       : len(STOI),
        "pad_id"           : int(PAD_ID),
        "source_map"       : SOURCE_MAP,
        "source_counts"    : source_counts,
        "cdr_statistics"   : {
            "samples_with_cdr3"    : int(cdr_stats["cdr3"]),
            "samples_with_any_cdr" : int(cdr_stats["any_cdr"]),
        },
        "seed"             : args.seed,
        "has_full_cdr_masks": True,
        "target_total"     : args.target_total,
        # STEP-3 new fields
        "has_clono_hash"   : True,
        "has_cdr3_len"     : True,
        "has_heavy_len"    : True,
        "hard_filters"     : {
            "cdr3_min_len"    : CDR3_MIN_LEN,
            "cdr3_max_len"    : CDR3_MAX_LEN,
            "min_cdr_tokens"  : MIN_CDR_TOKENS,
            "standard_aa_only": True,
            "v_gene_check"    : True,
        },
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("meta.json written")
    
    split_mmap_to_train_val(
        args.out_dir,
        seed=args.seed,
        val_ratio_hl=0.005,
        val_ratio_paired=0.005,
        workers=args.workers,
    )

    log("\n" + "=" * 60)
    log(f"Done!  Total written : {total_written:,}")
    log(f"  Paired : {source_counts['paired']:,}")
    log(f"  Heavy  : {source_counts['heavy']:,}")
    log(f"  Light  : {source_counts['light']:,}")
    log(f"  CDR3 coverage : {100*cdr_stats['cdr3']/max(total_written,1):.1f}%")
    log(f"  Any CDR       : {100*cdr_stats['any_cdr']/max(total_written,1):.1f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()

