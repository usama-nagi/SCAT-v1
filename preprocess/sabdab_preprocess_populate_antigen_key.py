#!/usr/bin/env python3
"""
assign_antigen_clusters.py
==========================
Pass-2 script: adds `antigen_cluster_id` to existing SAbDab .pt files
by clustering all antigen sequences with MMseqs2 at 70% identity.

Workflow
--------
1. Scan every .pt file in sabdab_dir, extract antigen_seq.
2. Write a FASTA file (one entry per unique antigen sequence).
3. Run MMseqs2 easy-cluster.
4. Parse the cluster_rep TSV -> dict {seq_hash: cluster_rep_id}.
5. Patch each .pt file in-place: add sample["antigen_cluster_id"].
6. Files without antigen (has_antigen=False) get antigen_cluster_id=None.

Prerequisites
-------------
    conda install -c bioconda mmseqs2
    # or
    pip install mmseqs2   (wraps the binary)

Usage
-----
    python assign_antigen_clusters.py \
        --sabdab_dir /srv/nfs-data/janeway/scat_data/processed/sabdab_v3_updated \
        --out_dir    /srv/nfs-data/janeway/scat_data/processed/sabdab_final \
        --tmp_dir    /tmp/mmseqs2_antigen \
        --min_seq_id 0.70 \
        --coverage   0.80 \
        --n_threads  16

    # If sabdab_dir == out_dir, files are patched in-place.
    # If out_dir is different, patched copies are written there (safer).

Output
------
Each .pt file gains:
    sample["antigen_cluster_id"] = "cluster_rep_<sha8>"  # or None if no antigen

The field is the SHA-8 prefix of the representative antigen sequence.
Identical or near-identical antigens get the same cluster_rep ID.
sabdab_loader.py's epitope_key logic will pick this up automatically
(it is the first priority key, before "target" string).
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch


# ─── helpers ──────────────────────────────────────────────────────────────────

def sha8(seq: str) -> str:
    """8-char SHA-256 prefix - short unique ID for a sequence string."""
    return hashlib.sha256(seq.encode()).hexdigest()[:8]


def check_mmseqs2():
    """Verify mmseqs2 binary is available."""
    # Run with no args - 'mmseqs --version' is not valid in all builds.
    # The binary always prints 'MMseqs2 Version: ...' on stdout/stderr
    # regardless of exit code, so we just check the output contains it.
    result = subprocess.run(["mmseqs"], capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if "MMseqs2" not in combined:
        print("[ERROR] mmseqs2 not found. Get the static binary:")
        print("  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz")
        print("  tar xzf mmseqs-linux-avx2.tar.gz")
        print("  export PATH=$(pwd)/mmseqs/bin/:$PATH")
        sys.exit(1)
    # Extract version line for logging
    for line in combined.splitlines():
        if "MMseqs2 Version" in line or "MMseqs2 (" in line:
            print(f"[mmseqs2] {line.strip()}")
            break


def scan_pt_files(sabdab_dir: str):
    """
    Scan .pt files. Return:
        file_paths      : list of str
        antigen_seqs    : list of str (empty string if no antigen)
        has_antigen     : list of bool
    """
    pt_files = sorted(Path(sabdab_dir).glob("*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files found in {sabdab_dir}")
        sys.exit(1)
    print(f"[scan] Found {len(pt_files)} .pt files in {sabdab_dir}")

    file_paths   = []
    antigen_seqs = []
    has_antigen  = []
    n_errors     = 0

    for i, pt in enumerate(pt_files):
        if i % 2000 == 0 and i > 0:
            print(f"  … scanned {i}/{len(pt_files)}", flush=True)
        try:
            s = torch.load(str(pt), map_location="cpu", weights_only=False)
        except Exception as e:
            n_errors += 1
            continue

        file_paths.append(str(pt))
        ag_seq  = s.get("antigen_seq", "") or ""
        has_ag  = bool(s.get("has_antigen", False)) and len(ag_seq) > 0
        antigen_seqs.append(ag_seq.strip() if has_ag else "")
        has_antigen.append(has_ag)

    print(f"  Loaded: {len(file_paths)}  |  with antigen: {sum(has_antigen)}  "
          f"|  errors: {n_errors}")
    return file_paths, antigen_seqs, has_antigen


def write_fasta(antigen_seqs, has_antigen, fasta_path: str):
    """
    Write one FASTA entry per UNIQUE antigen sequence.
    Entry ID = sha8(seq) so we can look up by content hash.
    Returns dict: {sha8: sequence} for all unique sequences.
    """
    unique_seqs = {}
    for seq, has in zip(antigen_seqs, has_antigen):
        if not has or not seq:
            continue
        h = sha8(seq)
        if h not in unique_seqs:
            unique_seqs[h] = seq

    print(f"[fasta] Unique antigen sequences: {len(unique_seqs)}")
    with open(fasta_path, "w") as f:
        for h, seq in unique_seqs.items():
            f.write(f">{h}\n{seq}\n")

    return unique_seqs


def run_mmseqs2(fasta_path: str, tmp_dir: str, min_seq_id: float,
                coverage: float, n_threads: int) -> str:
    """
    Run MMseqs2 easy-cluster. Returns path to cluster TSV
    (columns: rep_seq_id <TAB> member_seq_id).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    prefix = os.path.join(tmp_dir, "clusterDB")
    mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")

    cmd = [
        "mmseqs", "easy-cluster",
        fasta_path,
        prefix,
        mmseqs_tmp,
        "--min-seq-id", str(min_seq_id),
        "--cov-mode",   "0",          # bidirectional coverage
        "-c",           str(coverage),
        "--threads",    str(n_threads),
        "-v",           "2",
    ]
    print(f"[mmseqs2] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] MMseqs2 failed (exit {result.returncode})")
        sys.exit(1)

    tsv_path = prefix + "_cluster.tsv"
    if not os.path.exists(tsv_path):
        print(f"[ERROR] Expected output not found: {tsv_path}")
        sys.exit(1)
    print(f"[mmseqs2] Cluster TSV: {tsv_path}")
    return tsv_path


def parse_cluster_tsv(tsv_path: str) -> dict:
    """
    Parse MMseqs2 cluster TSV.
    Returns: {member_sha8: rep_sha8}
    Columns: rep_id <TAB> member_id
    """
    member_to_rep = {}
    n_lines = 0
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            member_to_rep[member] = rep
            n_lines += 1

    n_clusters = len(set(member_to_rep.values()))
    print(f"[cluster] Parsed {n_lines} member->rep mappings "
          f"across {n_clusters} clusters")
    return member_to_rep


def patch_pt_files(file_paths, antigen_seqs, has_antigen,
                   member_to_rep, out_dir, in_place: bool):
    """
    Patch each .pt file with antigen_cluster_id.
    If in_place: overwrite original. Else: write to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_patched    = 0
    n_with_clust = 0
    n_no_antigen = 0
    n_no_match   = 0
    n_errors     = 0

    for i, (pt_path, ag_seq, has_ag) in enumerate(
            zip(file_paths, antigen_seqs, has_antigen)):
        if i % 2000 == 0 and i > 0:
            print(f"  … patched {i}/{len(file_paths)} "
                  f"(with_clust={n_with_clust}, no_antigen={n_no_antigen}, "
                  f"no_match={n_no_match})", flush=True)
        try:
            s = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            n_errors += 1
            continue

        if not has_ag or not ag_seq:
            cluster_id = None
            n_no_antigen += 1
        else:
            h = sha8(ag_seq)
            rep = member_to_rep.get(h)
            if rep is not None:
                cluster_id = f"cluster_rep_{rep}"
                n_with_clust += 1
            else:
                # Should not happen - every sequence in FASTA is in the TSV
                # Fallback: use own sha8 as singleton cluster
                cluster_id = f"cluster_rep_{h}"
                n_no_match += 1

        s["antigen_cluster_id"] = cluster_id

        if in_place:
            dest = pt_path
        else:
            dest = os.path.join(out_dir, os.path.basename(pt_path))

        torch.save(s, dest)
        n_patched += 1

    print(f"\n[patch] Done.")
    print(f"  patched:          {n_patched}")
    print(f"  with cluster_id:  {n_with_clust}")
    print(f"  no antigen:       {n_no_antigen}")
    print(f"  no match (->own): {n_no_match}")
    print(f"  errors:           {n_errors}")

    # Cluster quality report
    if n_with_clust > 0:
        from collections import Counter
        # Count how many unique cluster IDs exist across files with antigen
        cluster_counter = Counter()
        for i, (pt_path, ag_seq, has_ag) in enumerate(
                zip(file_paths, antigen_seqs, has_antigen)):
            if not has_ag or not ag_seq:
                continue
            h   = sha8(ag_seq)
            rep = member_to_rep.get(h, h)
            cluster_counter[f"cluster_rep_{rep}"] += 1

        n_unique_clusters = len(cluster_counter)
        largest = cluster_counter.most_common(10)
        print(f"\n  CLUSTER QUALITY:")
        print(f"    Unique antigen clusters: {n_unique_clusters}")
        print(f"    (vs {sum(has_antigen)} files with antigen)")
        print(f"    Compression ratio: {sum(has_antigen)/max(n_unique_clusters,1):.1f}x")
        print(f"\n    Top 10 largest clusters:")
        for cid, cnt in largest:
            print(f"      {cid}  ->  {cnt} samples")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Assign MMseqs2 antigen cluster IDs to SAbDab .pt files"
    )
    parser.add_argument(
        "--sabdab_dir",  required=True,
        help="Directory containing preprocessed .pt files (input)"
    )
    parser.add_argument(
        "--out_dir",     default=None,
        help="Output directory for patched .pt files. "
             "If omitted, patches files in-place in sabdab_dir. "
             "Using a separate out_dir is safer for the first run."
    )
    parser.add_argument(
        "--tmp_dir",     default="/tmp/mmseqs2_antigen",
        help="Temp directory for MMseqs2 intermediate files"
    )
    parser.add_argument(
        "--min_seq_id",  type=float, default=0.70,
        help="MMseqs2 minimum sequence identity for clustering (default: 0.70)"
    )
    parser.add_argument(
        "--coverage",    type=float, default=0.80,
        help="MMseqs2 bidirectional coverage threshold (default: 0.80)"
    )
    parser.add_argument(
        "--n_threads",   type=int,   default=16,
        help="Number of MMseqs2 threads (default: 16)"
    )
    parser.add_argument(
        "--dry_run",     action="store_true",
        help="Run all steps except the final patching of .pt files"
    )
    parser.add_argument(
        "--skip_mmseqs", action="store_true",
        help="Skip MMseqs2 (use existing TSV). Requires --tsv_path."
    )
    parser.add_argument(
        "--tsv_path",    default=None,
        help="Path to existing MMseqs2 cluster TSV (for --skip_mmseqs)"
    )
    args = parser.parse_args()

    in_place = (args.out_dir is None or
                os.path.realpath(args.out_dir) == os.path.realpath(args.sabdab_dir))
    out_dir = args.sabdab_dir if in_place else args.out_dir

    print("=" * 60)
    print(f"  sabdab_dir : {args.sabdab_dir}")
    print(f"  out_dir    : {out_dir}  ({'in-place' if in_place else 'copy'})")
    print(f"  tmp_dir    : {args.tmp_dir}")
    print(f"  min_seq_id : {args.min_seq_id}")
    print(f"  coverage   : {args.coverage}")
    print(f"  n_threads  : {args.n_threads}")
    print("=" * 60)

    # Step 1 - check mmseqs2
    if not args.skip_mmseqs:
        check_mmseqs2()

    # Step 2 - scan .pt files
    print("\n[Step 1] Scanning .pt files …")
    file_paths, antigen_seqs, has_antigen = scan_pt_files(args.sabdab_dir)

    # Step 3 - write FASTA
    os.makedirs(args.tmp_dir, exist_ok=True)
    fasta_path = os.path.join(args.tmp_dir, "antigen_seqs.fasta")
    print(f"\n[Step 2] Writing FASTA -> {fasta_path} …")
    unique_seqs = write_fasta(antigen_seqs, has_antigen, fasta_path)

    if len(unique_seqs) == 0:
        print("[ERROR] No antigen sequences found. "
              "Check that .pt files contain 'antigen_seq' field.")
        sys.exit(1)

    # Step 4 - run MMseqs2
    if args.skip_mmseqs:
        if not args.tsv_path or not os.path.exists(args.tsv_path):
            print("[ERROR] --skip_mmseqs requires --tsv_path pointing to existing TSV")
            sys.exit(1)
        tsv_path = args.tsv_path
        print(f"\n[Step 3] Skipping MMseqs2, using existing TSV: {tsv_path}")
    else:
        print(f"\n[Step 3] Running MMseqs2 …")
        tsv_path = run_mmseqs2(
            fasta_path, args.tmp_dir,
            args.min_seq_id, args.coverage, args.n_threads
        )

    # Step 5 - parse cluster TSV
    print(f"\n[Step 4] Parsing cluster TSV …")
    member_to_rep = parse_cluster_tsv(tsv_path)

    # Step 6 - patch .pt files
    if args.dry_run:
        print("\n[Step 5] DRY RUN - skipping .pt patching.")
        print("  Re-run without --dry_run to apply changes.")
        return

    print(f"\n[Step 5] Patching .pt files …")
    if not in_place:
        # Copy unmodified files first (for files that fail to load etc)
        print(f"  Copying all files to {out_dir} first …")
        os.makedirs(out_dir, exist_ok=True)
        for pt in Path(args.sabdab_dir).glob("*.pt"):
            dest = Path(out_dir) / pt.name
            if not dest.exists():
                shutil.copy2(str(pt), str(dest))

    patch_pt_files(file_paths, antigen_seqs, has_antigen,
                   member_to_rep, out_dir, in_place)

    print(f"\n[Done] All .pt files patched with antigen_cluster_id.")

if __name__ == "__main__":
    main()
