#!/usr/bin/env python3

import argparse
import csv
import gzip
import logging
from collections import defaultdict
from pathlib import Path
import torch
from multiprocessing import Pool, cpu_count

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    filename="logs/oas_light.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# -------------------------
# Helpers
# -------------------------
def is_valid_aa(seq, min_len):
    if not isinstance(seq, str):
        return False
    if len(seq) < min_len:
        return False
    return seq.isalpha()


def pick_aa_sequence(row):
    seq = row.get("sequence_alignment_aa")
    if is_valid_aa(seq, 1):
        return seq.upper()
    seq = row.get("sequence_aa")
    if is_valid_aa(seq, 1):
        return seq.upper()
    return None


def seq_passes_basic_checks(seq):
    if not seq:
        return False
    if not set(seq).issubset(VALID_AA):
        return False
    if len(seq) > 160:
        return False
    for aa in VALID_AA:
        if seq.count(aa) / len(seq) > 0.25:
            return False
    return True


# -------------------------
# Validation
# -------------------------
def validate_oas_row(row, cfg, stats, chain):
    flags = {}

    aa = pick_aa_sequence(row)
    if not aa or len(aa) < cfg.min_seq_len or not seq_passes_basic_checks(aa):
        stats["bad_seq"] += 1
        return False, flags

    if chain == "Heavy" and not (90 <= len(aa) <= 150):
        stats["len_out_of_range"] += 1
        return False, flags

    if str(row.get("stop_codon", "")).upper() == "T":
        stats["stop_codon"] += 1
        return False, flags

    cdr3 = (row.get("cdr3_aa") or "").upper()
    if not cdr3:
        stats["missing_cdr3"] += 1
        return False, flags
    if not (cfg.min_cdr3 <= len(cdr3) <= cfg.max_cdr3):
        stats["cdr3_len_issue"] += 1
        return False, flags

    if str(row.get("productive", "")).upper() != "T":
        stats["not_productive"] += 1
        return False, flags

    if str(row.get("vj_in_frame", "")).upper() != "T":
        stats["out_of_frame"] += 1
        return False, flags

    anarci = str(row.get("ANARCI_status", ""))
    if anarci:
        flags["anarci_warning"] = True
        if "Missing conserved cysteine" in anarci:
            stats["severe_anarci"] += 1
            return False, flags
        if "Shorter than IMGT" in anarci:
            flags["imgt_short"] = True

    stats["kept"] += 1
    return True, flags


# -------------------------
# File filtering
# -------------------------
def keep_oas_file(path):
    name = path.name.upper()

    if "LIGHT" in name:
        return True
    if "IGHG" in name or "IGHA" in name:
        return True
    if "IGHE" in name or "BULK" in name:
        return False
    return False


# -------------------------
# File processing
# -------------------------
def process_oas_file(path, out_root, cfg):
    file_id = path.stem.replace(".csv", "")
    stats = defaultdict(int)

    if "Heavy" in file_id:
        chain = "Heavy"
    elif "Light" in file_id:
        chain = "Light"
    else:
        stats["unknown_chain"] += 1
        return stats

    out_dir = Path(out_root) / chain.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[PROCESS] {file_id} ({chain})")

    buffer = []
    shard_idx = 0

    with gzip.open(path, "rt", newline="") as fh:
        first_line = fh.readline().strip()
        if first_line.startswith("{") and first_line.endswith("}"):
            logger.info(f"[META] Skipped JSON header in {file_id}")

        reader = csv.DictReader(fh)

        for row in reader:
            stats["seen"] += 1
            keep, flags = validate_oas_row(row, cfg, stats, chain)
            if not keep:
                stats["skipped"] += 1
                continue

            sample = {
                "source": "OAS",
                "paired": False,
                "chain": chain,
                "sequence_aa": pick_aa_sequence(row),
                # "sequence_nt": row.get("sequence"),
                "cdr1_aa": (row.get("cdr1_aa") or "").upper(),
                "cdr2_aa": (row.get("cdr2_aa") or "").upper(),
                "cdr3_aa": (row.get("cdr3_aa") or "").upper(),
                "anarci_numbering": (row.get("ANARCI_numbering") or ""),
                "v_call": row.get("v_call"),
                "d_call": row.get("d_call"),
                "j_call": row.get("j_call"),
                "locus": row.get("locus"),
                "study_id": file_id,
                "quality_flags": flags,
            }

            buffer.append(sample)

            if len(buffer) >= cfg.max_per_file:
                out_path = out_dir / f"{file_id}_{shard_idx:04d}.pt"
                torch.save(buffer, out_path)
                logger.info(
                    f"[WRITE] {out_path.name} samples={len(buffer)}"
                )
                buffer.clear()
                shard_idx += 1

    if buffer:
        out_path = out_dir / f"{file_id}_{shard_idx:04d}.pt"
        torch.save(buffer, out_path)
        logger.info(
            f"[WRITE] {out_path.name} samples={len(buffer)}"
        )

    return stats


# -------------------------
# Multiprocessing worker
# -------------------------
def _worker(args):
    path, out_root, cfg = args
    stats = process_oas_file(path, out_root, cfg)
    logger.info(
        f"[DONE] {path.name}: seen={stats['seen']} kept={stats['kept']} skipped={stats['skipped']}"
    )
    return stats


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--min_seq_len", type=int, default=10)
    parser.add_argument("--min_cdr3", type=int, default=5)
    parser.add_argument("--max_cdr3", type=int, default=40)
    parser.add_argument("--max-per-file", type=int, default=50_000)
    args = parser.parse_args()

    all_files = sorted(Path(args.raw_root).glob("*.csv.gz"))
    files = [f for f in all_files if keep_oas_file(f)]

    if args.max_files:
        files = files[: args.max_files]

    # sort by compressed size, biggest first
    files.sort(key=lambda p: p.stat().st_size, reverse=True)

    logger.info(
                "[SCHEDULER] Files sorted by size (largest first)")

    logger.info(f"[START] Found {len(files)} files")

    n_workers = min(24, cpu_count() - 2)
    logger.info(f"[PARALLEL] Using {n_workers} workers")

    with Pool(n_workers) as pool:
        results = pool.map(
            _worker,
            [(p, args.out_root, args) for p in files]
        )

    total_kept = sum(r["kept"] for r in results)
    logger.info(f"[SUMMARY] Total sequences written: {total_kept}")


if __name__ == "__main__":
    main()
