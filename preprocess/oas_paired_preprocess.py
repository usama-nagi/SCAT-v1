#!/usr/bin/env python3

import argparse
import gzip
import json
import logging
from pathlib import Path

import pandas as pd
import torch


# ------------------------
# Logging
# ------------------------

logging.basicConfig(
    filename="logs/oas_process_paired.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------
# Safe helpers
# ------------------------

def safe_get(row, key, default=None):
    try:
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


def is_valid_aa(seq, min_len=5):
    if not isinstance(seq, str):
        return False
    if len(seq) < min_len:
        return False
    return seq.isalpha()


def pick_aa_sequence(row, prefix):
    seq = safe_get(row, f"sequence_alignment_aa_{prefix}")
    if is_valid_aa(seq):
        return seq

    seq = safe_get(row, f"sequence_aa_{prefix}")
    if is_valid_aa(seq):
        return seq

    return None


# ------------------------
# Validation logic
# ------------------------

def validate_chain(row, prefix):
    seq = pick_aa_sequence(row, prefix)
    cdr1 = safe_get(row, f"cdr1_aa_{prefix}")
    cdr2 = safe_get(row, f"cdr2_aa_{prefix}")
    cdr3 = safe_get(row, f"cdr3_aa_{prefix}")
    productive = safe_get(row, f"productive_{prefix}")
    vj_in_frame = safe_get(row, f"vj_in_frame_{prefix}")

    flags = {
        "has_sequence": is_valid_aa(seq, min_len=10),
        "has_cdr3": is_valid_aa(cdr3, min_len=5),
        "productive": productive in ("T", True, "True", 1),
        "vj_in_frame": vj_in_frame in ("T", True, "True", 1),
    }

    ok = all(flags.values())
    return ok, flags, seq, cdr1, cdr2, cdr3


# ------------------------
# File processing with sharding
# ------------------------

def process_file(csv_gz_path, out_dir, buffer, shard_idx, max_per_file):
    """
    Process a single CSV file and accumulate samples into buffer.
    Returns: (kept, skipped, updated_buffer, updated_shard_idx)
    """
    kept = 0
    skipped = 0

    with gzip.open(csv_gz_path, "rt") as f:
        first_line = f.readline().strip()
        if first_line.startswith('"') and first_line.endswith('"'):
            first_line = first_line[1:-1]
        fixed_line = first_line.replace('""', '"')

        try:
            study_meta = json.loads(fixed_line)
        except Exception:
            logger.warning(f"[WARN] No JSON header in {csv_gz_path.name}")
            study_meta = {}

        df = pd.read_csv(f)

    for _, row in df.iterrows():
        heavy_ok, heavy_flags, heavy_seq, heavy_cdr1, heavy_cdr2, heavy_cdr3 = validate_chain(row, "heavy")
        light_ok, light_flags, light_seq, light_cdr1, light_cdr2, light_cdr3 = validate_chain(row, "light")

        if not (heavy_ok and light_ok):
            skipped += 1
            continue

        sample = {
            "source": "OAS_paired",
            "paired": True,

            "study": {
                "run": study_meta.get("Run"),
                "author": study_meta.get("Author"),
                "species": study_meta.get("Species"),
                "subject": study_meta.get("Subject"),
                "disease": study_meta.get("Disease"),
            },

            "heavy": {
                "sequence_aa": heavy_seq,
                "cdr1_aa": heavy_cdr1,
                "cdr2_aa": heavy_cdr2,
                "cdr3_aa": heavy_cdr3,
                "v_call": safe_get(row, "v_call_heavy"),
                "j_call": safe_get(row, "j_call_heavy"),
                "locus": safe_get(row, "locus_heavy"),
            },

            "light": {
                "sequence_aa": light_seq,
                "cdr1_aa": light_cdr1,
                "cdr2_aa": light_cdr2,
                "cdr3_aa": light_cdr3,
                "v_call": safe_get(row, "v_call_light"),
                "j_call": safe_get(row, "j_call_light"),
                "locus": safe_get(row, "locus_light"),
            },

            "anarci": {
                "heavy": safe_get(row, "ANARCI_numbering_heavy"),
                "light": safe_get(row, "ANARCI_numbering_light"),
            },

            "lengths": {
                "heavy": len(heavy_seq),
                "light": len(light_seq),
                "cdr3_heavy": len(heavy_cdr3),
                "cdr3_light": len(light_cdr3),
            },

            "quality_flags": {
                "heavy": heavy_flags,
                "light": light_flags,
                "has_anarci": bool(
                    safe_get(row, "ANARCI_numbering_heavy")
                    or safe_get(row, "ANARCI_numbering_light")
                ),
            },
        }

        buffer.append(sample)
        kept += 1

        # Write shard when buffer is full
        if len(buffer) >= max_per_file:
            out_path = out_dir / f"paired_{shard_idx:04d}.pt"
            torch.save(buffer, out_path)
            logger.info(
                f"[WRITE] {out_path.name} samples={len(buffer)}"
            )
            buffer.clear()
            shard_idx += 1

    return kept, skipped, buffer, shard_idx


# ------------------------
# Entry point
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max-per-file", type=int, default=50_000)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_skipped = 0
    
    # Shared state across all files
    buffer = []
    shard_idx = 0

    files = sorted(input_dir.glob("*.csv.gz"))

    for i, fpath in enumerate(files, 1):
        logger.info(f"[FILE {i}/{len(files)}] {fpath.name}")

        kept, skipped, buffer, shard_idx = process_file(
            fpath,
            output_dir,
            buffer,
            shard_idx,
            args.max_per_file,
        )

        total_kept += kept
        total_skipped += skipped

        logger.info(
            f"[DONE] {fpath.stem}: kept={kept}, skipped={skipped}, "
            f"buffer_size={len(buffer)}, next_shard={shard_idx}"
        )

    # Write remaining buffer
    if buffer:
        out_path = output_dir / f"paired_{shard_idx:04d}.pt"
        torch.save(buffer, out_path)
        logger.info(
            f"[WRITE] {out_path.name} samples={len(buffer)} (final)"
        )

    logger.info(
        f"[SUMMARY] kept={total_kept}, skipped={total_skipped}, "
        f"total_shards={shard_idx + 1}"
    )


if __name__ == "__main__":
    main()
