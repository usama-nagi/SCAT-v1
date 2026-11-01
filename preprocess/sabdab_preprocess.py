"""
preprocess_sabdab.py
====================
Download the full structural SAbDab, parse each PDB, and save per-antibody
.pt files that are drop-in compatible with the existing SAbDabDataset loader.

Usage
-----
# 1. Download summary CSV from https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab
#    Click "Search" with default filters -> "Download summary"

# 2. Download structures + IMGT annotations
python sabdab_downloader.py -s summary.tsv -o /raw_sabdab/ --chothia_pdb --imgt

# 3. Run this script
python scripts/preprocess/sabdab_preprocess.py \
    --summary    /srv/nfs-data/janeway/scat_data/raw_data/sabdab_summary_all.tsv \
    --struct_dir /srv/nfs-data/janeway/scat_data/raw_data/all_structures \
    --out_dir    /srv/nfs-data/janeway/scat_data/processed/sabdab_v2 \
    --n_workers  8

Output
------
One .pt file per antibody chain pair with keys matching what SAbDabDataset expects:
    heavy / light / coords / coord_mask / cdr_mask / interface_mask /
    antigen_coords / antigen_mask / source / paired / task_mask
"""

import os
import csv
import argparse
import warnings
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

#  Dependencies 
# pip install biopython
try:
    from Bio import PDB
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    raise ImportError("pip install biopython")

warnings.filterwarnings("ignore")

#  Constants 
IMGT_CDR_H = {
    "H1": (27,   38.9),   # IMGT 27-38, insertions at 31A,31B...
    "H2": (56,   65.9),   # IMGT 56-65, rarely has insertions
    "H3": (105,  117.9),  # IMGT 105-117, insertions at 111A... and 112A...
}

IMGT_CDR_L = {
    "L1": (27,   38.9),   # IMGT 27-38, insertions at 30A,30B...
    "L2": (56,   65.9),   # IMGT 56-65
    "L3": (105,  117.9),  # IMGT 105-117, insertions at 111A...
}

AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

INTERFACE_CUTOFF = 10.0   # A  - residues within this distance of antigen

_NULL_TOKENS = {"", "none", "n/a", "na", "nan", "unknown", "null"}

def _norm_target(s):
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    sl = s.lower()
    if sl in _NULL_TOKENS: return None
    parts = [p.strip() for p in sl.split("|") if p.strip()]
    return parts[0] if parts else None

#  PDB Parsing 

def _residue_aa(res) -> Optional[str]:
    """Return single-letter AA or None if not a standard residue."""
    if not is_aa(res, standard=True):
        return None
    name = res.get_resname().strip()
    return AA_MAP.get(name, None)


def _ca_coords(res) -> Optional[np.ndarray]:
    """Return Cα coordinates or None."""
    try:
        return np.array(res["CA"].get_vector().get_array(), dtype=np.float32)
    except KeyError:
        return None

def get_fv_residues(chain):
    residues = [r for r in chain.get_residues() if is_aa(r, standard=True)]
    residues.sort(key=lambda r: (r.get_id()[1], r.get_id()[2]))
    fv_residues = []
    for r in residues:
        if r.get_id()[1] <= 128:
            fv_residues.append(r)
    return fv_residues

def parse_chain(chain, is_heavy: bool):
    residues = get_fv_residues(chain)
    if not residues:
        return None, None, None, None, None

    seq        = []
    coords     = []
    coord_mask = []
    imgt_nums  = []

    for res in residues:
        aa = _residue_aa(res)
        if aa is None:
            continue

        seq.append(aa)

        ca = _ca_coords(res)
        if ca is not None:
            coords.append(ca)
            coord_mask.append(True)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            coord_mask.append(False)

        res_id = res.get_id()
        pos    = res_id[1]
        icode  = res_id[2].strip()
        # Encode insertion codes as fractional offsets:
        # 100A=100.1, 100B=100.2... preserves ordering and stays > 100
        if icode:
            pos = pos + (ord(icode.upper()) - ord('A') + 1) * 0.1
        imgt_nums.append(pos)


    seq        = "".join(seq)
    coords     = np.stack(coords)
    coord_mask = np.array(coord_mask)
    imgt_nums  = np.array(imgt_nums)
    
    # CDR mask using IMGT ranges
    cdr_ranges = IMGT_CDR_H if is_heavy else IMGT_CDR_L
    cdr_mask   = np.zeros(len(seq), dtype=bool)

    for (start, end) in cdr_ranges.values():
        cdr_mask |= (imgt_nums >= start) & (imgt_nums <= end)

    assert len(seq) == len(imgt_nums)

    # Keep only the Fv variable domain (IMGT positions 1-128).
    # Positions above 128 are constant domain / hinge - they have no CDR
    # annotations and add noise with zero useful supervision signal.
    # This uses the IMGT numbers already computed above, so no extra parsing.
    # FV_MAX_IMGT = 128
    FV_MAX_IMGT = 128.99  # include 128 + any insertion codes at 128
    fv_mask    = imgt_nums <= FV_MAX_IMGT
    seq        = "".join([seq[i]      for i in range(len(seq))      if fv_mask[i]])
    coords     = coords[fv_mask]
    coord_mask = coord_mask[fv_mask]
    cdr_mask   = cdr_mask[fv_mask]
    imgt_nums  = imgt_nums[fv_mask]

    if len(seq) < 10:   # guard: if almost nothing left, treat as missing
        return None, None, None, None, None

    return seq, coords, coord_mask, imgt_nums, cdr_mask

def compute_interface_mask(ab_residues, ag_residues, cutoff=INTERFACE_CUTOFF):
    """
    Flag antibody residues within `cutoff` A of any antigen atom.
    Returns np.ndarray[bool] of length len(ab_residues).
    """
    if not ag_residues:
        return np.zeros(len(ab_residues), dtype=bool)

    ag_atoms = [a for res in ag_residues for a in res.get_atoms()]
    if not ag_atoms:
        return np.zeros(len(ab_residues), dtype=bool)

    ns   = NeighborSearch(ag_atoms)
    mask = np.zeros(len(ab_residues), dtype=bool)

    for i, res in enumerate(ab_residues):
        for atom in res.get_atoms():
            if ns.search(atom.get_vector().get_array(), cutoff, "A"):
                mask[i] = True
                break

    return mask


#  Per-entry processing 

def process_entry(
        pdb_id,
        h_chain,
        l_chain,
        ag_chain,
        model_id,
        antigen_name,
        struct_dir,
        out_dir,
    ) -> dict:
    """
    Process one antibody entry. Returns a status dict for logging.
    """
    pdb_id   = pdb_id.lower()
    pdb_file = os.path.join(struct_dir, "imgt", f"{pdb_id}.pdb")

    if not os.path.isfile(pdb_file):
        return {"pdb": pdb_id, "status": "missing_pdb"}

    safe_pdb = pdb_id.replace("|", "_")
    safe_h = h_chain.replace("|", "_")
    safe_l = l_chain.replace("|", "_")
    safe_ag = ag_chain.replace("|", "_")
    safe_model = str(model_id).replace("|", "_")
    
    out_name = f"{safe_pdb}_{safe_h}{safe_l}_ag{safe_ag}_m{safe_model}.pt"
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return {"pdb": pdb_id, "status": "cached"}

    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = list(structure.get_models())

        model_idx = min(int(model_id), len(models) - 1)
        model = models[model_idx]
    except Exception as e:
        return {"pdb": pdb_id, "status": f"parse_error: {e}"}

    #  Parse heavy chain 
    if h_chain not in model:
        return {"pdb": pdb_id, "status": f"missing_H_chain_{h_chain}"}
    h_seq, h_coords, h_coord_mask, h_imgt, h_cdr = parse_chain(
    model[h_chain], is_heavy=True
)
    if h_seq is None or len(h_seq) < 10:
        return {"pdb": pdb_id, "status": "short_H"}

    #  Parse light chain 
    l_seq = ""
    l_coords = None
    l_coord_mask = None
    l_cdr = None
    has_light = l_chain != "NA" and l_chain in model
    if has_light:
        l_seq, l_coords, l_coord_mask, l_imgt, l_cdr = parse_chain(
            model[l_chain], is_heavy=False
        )
        if l_seq is None or len(l_seq) < 10:
            has_light = False

    #  Concatenate heavy + light 
    if has_light:
        seq        = h_seq + l_seq
        coords     = np.concatenate([h_coords,     l_coords],     axis=0)  # [L, 3]
        coord_mask = np.concatenate([h_coord_mask, l_coord_mask], axis=0)  # [L]
        cdr_mask   = np.concatenate([h_cdr,        l_cdr],        axis=0)  # [L]
    else:
        seq        = h_seq
        coords     = h_coords
        coord_mask = h_coord_mask
        cdr_mask   = h_cdr

    if len(seq) > 280:
        # Safety cap - should not trigger after Fv filtering above
        seq        = seq[:280]
        coords     = coords[:280]
        coord_mask = coord_mask[:280]
        cdr_mask   = cdr_mask[:280]

    #  Identify antigen chains 
    ab_chain_ids = {str(h_chain).strip().upper()}
    if has_light:
        ab_chain_ids.add(str(l_chain).strip().upper())
    
    if ag_chain and ag_chain.upper() != "NA":
        requested_ag = {x.strip().upper() for x in ag_chain.split("|") if x.strip()}
        # NEW: reject any antigen chain that is actually an antibody chain.
        # SAbDab summary occasionally lists the antibody itself in antigen_chain.
        requested_ag = requested_ag - ab_chain_ids
        if not requested_ag:
            ag_chains = []                      # no real antigen -> drop
        else:
            ag_chains = [c for c in model.get_chains()
                        if str(c.id).strip().upper() in requested_ag]
    else:
        ag_chains = []
    
    ag_residues = [r for c in ag_chains
                   for r in c.get_residues() if is_aa(r, standard=True)]

    has_antigen = len(ag_residues) > 0

    #  Antigen coordinates 
    antigen_coords = None
    antigen_mask   = None
    antigen_seq = None
    if has_antigen:
        ag_coords_list = []
        ag_mask_list   = []
        for res in ag_residues[:2000]:   # cap antigen length
            ca = _ca_coords(res)
            if ca is not None:
                ag_coords_list.append(ca)
                ag_mask_list.append(True)
            else:
                ag_coords_list.append(np.zeros(3, dtype=np.float32))
                ag_mask_list.append(False)
        antigen_coords = torch.tensor(np.stack(ag_coords_list), dtype=torch.float32)
        antigen_mask   = torch.tensor(ag_mask_list, dtype=torch.bool)
        # store antigen residue sequence for chemistry conditioning
        antigen_seq    = "".join(
            AA_MAP.get(res.get_resname().strip(), "X")
            for res in ag_residues[:2000]
        )

    #  Interface mask 
    interface_mask = None
    if has_antigen:
        ab_residues_list = []
        for chain_id in ([h_chain] + ([l_chain] if has_light else [])):
            if chain_id in model:
                ab_residues_list.extend(get_fv_residues(model[chain_id]))
        ab_residues_list = ab_residues_list[:2000]
        iface_np       = compute_interface_mask(ab_residues_list, ag_residues)
        iface_np       = iface_np[:len(seq)]   # align to truncated length
        interface_mask = torch.tensor(iface_np, dtype=torch.bool)

    #  Validity checks before saving 
    # Need at least 3 valid Cα residues for Kabsch
    if coord_mask.sum() < 3:
        return {"pdb": pdb_id, "status": "too_few_coords"}
    # Need at least one CDR position
    if cdr_mask.sum() == 0:
        return {"pdb": pdb_id, "status": "no_cdr_positions",
        "n_cdr_h": int(h_cdr.sum()), "seq_len": len(seq)}

    #  Build output dict 
    coords_t     = torch.tensor(coords,     dtype=torch.float32)   # [L, 3]
    coord_mask_t = torch.tensor(coord_mask, dtype=torch.bool)      # [L]
    cdr_mask_t   = torch.tensor(cdr_mask,   dtype=torch.bool)      # [L]
    
    if h_seq is not None and h_cdr.sum() > 0:
        padded = np.concatenate(([0], h_cdr.astype(np.uint8), [0]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]
        # H3 is the 3rd block (index 2) in IMGT layout H1,H2,H3
        if len(starts) >= 3:
            h3_len = int(ends[2] - starts[2])
        elif len(starts) > 0:
            h3_len = int(ends[-1] - starts[-1])  # fallback: last block
        else:
            h3_len = 0
    else:
        h3_len = 0
    
    if h3_len < 4:
        return {"pdb": pdb_id, "status": f"h3_too_short_{h3_len}"}
    if h3_len > 36:
        return {"pdb": pdb_id, "status": f"h3_too_long_{h3_len}"}

    sample = {
        "source":    "SAbDab",
        "pdb_id":    pdb_id,
        "paired":    has_light,
        "therapeutic": False,
        "target":    antigen_name,
        "heavy": {"sequence_aa": h_seq},
        "light": {"sequence_aa": l_seq if has_light else ""},
        "coords":           coords_t,
        "coord_mask":       coord_mask_t,
        "antigen_coords":   antigen_coords,
        "antigen_mask":     antigen_mask,
        "antigen_seq":      antigen_seq if has_antigen else "",
        "cdr_mask":         cdr_mask_t,
        "cdr3_len_actual":  h3_len,
        "interface_mask":   interface_mask,
        "has_antigen":      has_antigen,
        "antigen_cluster_id": None,
        "task_mask": {
            "AR":  True,
            "MLM": False,
            "GEO": True,                     # always True - we have coords
            "IF":  has_antigen,              # True only when antigen present
        },
    }

    torch.save(sample, out_path)
    return {"pdb": pdb_id, "status": "ok", "has_antigen": has_antigen,
            "seq_len": len(seq), "n_cdr": int(cdr_mask.sum())}


#  Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary",    required=True,
                        help="Tab-separated summary CSV from SAbDab website")
    parser.add_argument("--struct_dir", required=True,
                        help="Root directory from sabdab_downloader (contains sabdab_dataset/)")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory for .pt files")
    parser.add_argument("--n_workers",  type=int, default=4)
    parser.add_argument("--max_entries", type=int, default=None,
                        help="Cap number of entries (for testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # struct_dir may point to the parent or directly to sabdab_dataset/
    struct_dir = args.struct_dir
    if os.path.isdir(os.path.join(struct_dir, "sabdab_dataset")):
        struct_dir = os.path.join(struct_dir, "sabdab_dataset")

    #  Parse summary file 
    entries = []
    with open(args.summary) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pdb = row.get("pdb", "").strip().lower()
            h   = row.get("Hchain", "").strip()
            l   = row.get("Lchain", "NA").strip()
            if not pdb or not h:
                continue
            ag = row.get("antigen_chain", "").strip()
            model = row.get("model", "0").strip()
            antigen_name = _norm_target(row.get("antigen_name", ""))
            entries.append((pdb, h, l, ag, model, antigen_name))

    if args.max_entries:
        entries = entries[:args.max_entries]

    print(f"[preprocess_sabdab] Found {len(entries)} entries in summary")
    print(f"[preprocess_sabdab] struct_dir: {struct_dir}")
    print(f"[preprocess_sabdab] out_dir:    {args.out_dir}")
    print(f"[preprocess_sabdab] n_workers:  {args.n_workers}")

    #  Process in parallel 
    stats = {"ok": 0, "cached": 0, "missing_pdb": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(
                process_entry,
                pdb,
                h,
                l,
                ag,
                model,
                antigen_name,
                struct_dir,
                args.out_dir,
            ): (pdb, h, l, ag, model, antigen_name)
            for pdb, h, l, ag, model, antigen_name in entries
        }
        for i, fut in enumerate(as_completed(futures)):
            try:
                result = fut.result()
            except Exception as e:
                pdb, h, l, ag, model, antigen_name = futures[fut]
                result = {"pdb": pdb, "status": f"exception: {e}"}

            status = result.get("status", "error")
            if status == "ok":
                stats["ok"] += 1
            elif status == "cached":
                stats["cached"] += 1
            elif status == "missing_pdb":
                stats["missing_pdb"] += 1
            else:
                stats["error"] += 1
                if stats["error"] <= 20:   # only print first 20 errors
                    print(f"  SKIP {result['pdb']}: {status}")

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(entries)}] ok={stats['ok']} "
                      f"cached={stats['cached']} skip={stats['missing_pdb']+stats['error']}")

    print(f"\n[preprocess_sabdab] Done.")
    print(f"  saved:       {stats['ok']}")
    print(f"  cached:      {stats['cached']}")
    print(f"  missing_pdb: {stats['missing_pdb']}")
    print(f"  errors:      {stats['error']}")
    print(f"  total .pt:   {len(list(Path(args.out_dir).glob('*.pt')))}")


if __name__ == "__main__":
    main()
