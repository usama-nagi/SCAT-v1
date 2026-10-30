# data/loaders/sabdab_loader.py
import os
import torch
from torch.utils.data import Dataset
from utils.vocab import STOI
import random

def _load_exclude_keys(txt_path: str) -> set:
    """
    Load exclusion keys from a test IDs file.
    Supports:
      '2adf_H_L_A'  -> normalized key '2adf_hl_aga'
      '2adf'        -> PDB-only fallback (excludes all entries for that PDB)
    Returns a set of normalized keys AND a set of fallback PDB IDs.
    """
    import re
    if not txt_path:
        return set(), set()
    
    if not os.path.isabs(txt_path):
        here = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.join(here, txt_path)
        if os.path.exists(resolved):
            txt_path = resolved
    
    if not os.path.exists(txt_path):
        print(f"[SAbDabDataset] WARNING: exclude file not found: {txt_path!r}")
        return set(), set()
    
    full_keys = set()   # normalized full keys like '2adf_hl_aga'
    pdb_only  = set()   # fallback 4-char IDs for entries without chain info

    with open(txt_path) as f:
        for line in f:
            pid = line.strip().split("#")[0].strip()
            if not pid:
                continue
            parts = pid.lower().split("_")
            pdb = parts[0]
            if not (len(pdb) == 4 and pdb.isalnum()):
                continue
            if len(parts) >= 4 and not parts[2].startswith("ag"):
                # Full format: pdb_H_L_AG -> '2adf_hl_aga'
                # (token[2] is a bare chain id here, e.g. 'l'/'a'/'k' - never 'ag*')
                heavy, light, ag = parts[1], parts[2], parts[3]
                chains = heavy + light
                key = f"{pdb}_{chains}_ag{ag}"
                # strip model suffix pattern if present
                key = re.sub(r'_m\d+$', '', key)
                full_keys.add(key)
            elif len(parts) >= 3 and parts[2].startswith("ag"):
                # Filename-shaped format: pdb_HL_agX[_mN] -> '1ar1_cd_agb'
                # (already matches _file_key_from_path's own key construction -
                # token[1] is the combined chain string, token[2] already has
                # the 'ag' prefix attached, e.g. 'agb', 'agna', 'ag')
                chains = parts[1]
                ag_token = parts[2]
                key = f"{pdb}_{chains}_{ag_token}"
                key = re.sub(r'_m\d+$', '', key)
                full_keys.add(key)
            else:
                # Only PDB ID available - exclude all entries for this PDB
                pdb_only.add(pdb)

    print(f"[SAbDabDataset] Exclude: {len(full_keys)} full keys + "
          f"{len(pdb_only)} PDB-only IDs from {txt_path}")
    return full_keys, pdb_only

def _load_exact_filenames(txt_path: str) -> set:
    """
    Load exact filenames for exclusion.

    Example:
        8vyf_HL_agA_m0.pt
        8vyg_AB_agR_m0.pt
    """
    if not txt_path:
        return set()

    if not os.path.isabs(txt_path):
        here = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.join(here, txt_path)
        if os.path.exists(resolved):
            txt_path = resolved

    if not os.path.exists(txt_path):
        print(f"[SAbDabDataset] WARNING: exclude file not found: {txt_path!r}")
        return set()

    files = set()

    with open(txt_path) as f:
        for line in f:
            name = line.strip().split("#")[0].strip()
            if name:
                files.add(name.lower())

    print(
        f"[SAbDabDataset] Exclude: {len(files)} exact filenames "
        f"from {txt_path}"
    )

    return files


def _file_key_from_path(path: str) -> tuple:
    """
    Return (normalized_full_key, pdb_id) from a .pt filename.
    '2adf_HL_agA_m0.pt' -> ('2adf_hl_aga', '2adf')
    """
    import re
    stem   = os.path.splitext(os.path.basename(path))[0].lower()
    key    = re.sub(r'_m\d+$', '', stem)   # strip _m0, _m1...
    pdb_id = stem.split("_")[0]
    return key, pdb_id

def _extract_h3_key(heavy_seq: str, cdr_mask, fallback_idx: int) -> str:
    """CDR-H3 amino-acid sequence used as clonotype key.
    For v2 IMGT data: heavy chain has 3 CDR blocks (H1, H2, H3).
    H3 is always block index 2 (third block) within the heavy chain portion."""
    try:
        heavy = heavy_seq or ""
        if cdr_mask is None or len(cdr_mask) == 0 or len(heavy) == 0:
            return f"__no_cdr_{fallback_idx}"
        heavy_len = len(heavy)
        # Only look at the heavy chain portion of the mask
        cdr_heavy = cdr_mask[:heavy_len].long()
        padded = torch.cat([
            torch.zeros(1, dtype=torch.long),
            cdr_heavy,
            torch.zeros(1, dtype=torch.long),
        ])
        diff   = padded[1:] - padded[:-1]
        starts = (diff ==  1).nonzero(as_tuple=False).flatten().tolist()
        ends   = (diff == -1).nonzero(as_tuple=False).flatten().tolist()
        if not starts or not ends:
            return f"__no_cdr_{fallback_idx}"
        # H3 is block index 2 (3-block IMGT layout: H1, H2, H3)
        # Fall back to last block if fewer than 3 blocks (legacy data)
        h3_idx = 2 if len(starts) >= 3 else len(starts) - 1
        h3 = heavy[starts[h3_idx]:ends[h3_idx]]
        return h3 if h3 else f"__no_h3_{fallback_idx}"
    except Exception:
        return f"__err_{fallback_idx}"

def _extract_h3_mask(heavy_seq: str, cdr_mask: torch.Tensor) -> torch.Tensor:
    """
    H3-only boolean mask over the full (heavy+light) sequence length,
    using the same block-boundary logic as _extract_h3_key() but returning
    a mask instead of a string. Light-chain positions are always False.
    """
    L_total = len(cdr_mask)
    h3_mask = torch.zeros(L_total, dtype=torch.bool)
    heavy_len = len(heavy_seq)
    if heavy_len == 0:
        return h3_mask
    cdr_heavy = cdr_mask[:heavy_len].long()
    padded = torch.cat([torch.zeros(1, dtype=torch.long), cdr_heavy, torch.zeros(1, dtype=torch.long)])
    diff   = padded[1:] - padded[:-1]
    starts = (diff ==  1).nonzero(as_tuple=False).flatten().tolist()
    ends   = (diff == -1).nonzero(as_tuple=False).flatten().tolist()
    if not starts or not ends:
        return h3_mask
    h3_idx = 2 if len(starts) >= 3 else len(starts) - 1
    h3_mask[starts[h3_idx]:ends[h3_idx]] = True
    return h3_mask

CONTACT_CUTOFF_ANGSTROM = 8.0

def _compute_contact_mask(coords, coord_mask, h3_mask, antigen_coords, antigen_mask,
                           contact_cutoff: float = CONTACT_CUTOFF_ANGSTROM):
    """
    Per-position antigen-contact mask, restricted to H3 positions ONLY -
    mirrors path_isolation_by_contact()'s `h3_b = cdr3_b & (chain == 0)`
    bucket definition exactly (H3 only, not L3 - eval never measures L3
    contact, so training must not upweight it under this signal either).

    coords:          [L, N, 3] antibody coords (N = atoms/residue, as stored
                      post-truncation in __getitem__; N=1 today via unsqueeze(1))
    h3_mask:         [L] bool, True at H3-only positions (cdr3 & heavy chain)
    antigen_coords:  [Lag, N, 3] antigen coords
    antigen_mask:    [Lag, N] bool, valid per atom-slot

    Returns: [L] float tensor, matches path_isolation_by_contact()'s contact
             criterion bit-for-bit: same Ca-index selection rule, same
             any-atom-slot antigen validity check, same strict '<' cutoff.
    """
    if coords is None or antigen_coords is None or h3_mask is None or h3_mask.sum() == 0:
        return None

    n_atoms = coords.shape[1]
    ca_idx  = 1 if n_atoms >= 4 else 0
    ab_ca   = coords[:, ca_idx, :].float()          # [L, 3]
    ag_ca   = antigen_coords[:, ca_idx, :].float()   # [Lag, 3]

    ag_valid = (antigen_mask.bool().any(dim=-1) if antigen_mask is not None
                else torch.ones(ag_ca.shape[0], dtype=torch.bool, device=ag_ca.device))
    ag_ca = ag_ca[ag_valid]

    L = coords.shape[0]
    contact = torch.zeros(L, dtype=torch.float32)
    if ag_ca.shape[0] == 0:
        return contact

    h3_idx = h3_mask.nonzero(as_tuple=True)[0]
    if h3_idx.numel() == 0:
        return contact

    dists = torch.cdist(ab_ca[h3_idx], ag_ca)        # [n_h3, n_ag_valid]
    min_dist = dists.min(dim=1).values
    is_contact = (min_dist < contact_cutoff)          # strict '<', matches eval

    contact[h3_idx] = is_contact.float()
    return contact

class SAbDabDataset(Dataset):
    """
    Loader for processed SAbDab .pt files with smart CDR-preserving truncation.
    
    Compatible with both:
      - TheraSAbDab (sequence-only, no coords)
      - Full structural SAbDab (coords + antigen + CDR + interface masks)

    The .pt files are produced by preprocess_sabdab.py and have this schema:
        heavy/light         : dict with sequence_aa
        coords              : [L, 3] float32 tensor or None
        coord_mask          : [L] bool tensor or None
        antigen_coords      : [Lag, 3] float32 tensor or None
        antigen_mask        : [Lag] bool tensor or None
        cdr_mask            : [L] bool tensor or None
        interface_mask      : [L] bool tensor or None
        task_mask           : dict (AR/MLM/GEO/IF)
        source / pdb_id / paired / therapeutic / target
    """

    def __init__(self, root_dir: str, max_seq_len: int = 280,
                 require_structure: bool = False,
                 require_antigen: bool = True,
                 exclude_ids: set = None, # type: ignore
                 agforce_exclude_file: str = "/home/abad/scat-mini/data/loaders/agforce_ids.txt",
                 h3_exclude_file: str = "/home/abad/scat-mini/data/loaders/agforce_exclude_ids.txt"):
        """
        Args:
            root_dir:          Directory containing .pt files
            max_seq_len:       Maximum sequence length (truncate with CDR preservation)
            require_structure: If True, skip samples without coordinates
            require_antigen:   If True, skip samples without antigen coords
        """
        self.root_dir = root_dir
        self.max_seq_len = max_seq_len
        
        _manual_ids = set(exclude_ids or [])
        _full_keys, _pdb_only = _load_exclude_keys(agforce_exclude_file)
        _pdb_only |= {i.lower()[:4] for i in _manual_ids if len(i) >= 4}
        
        # Near80 exclusions are exact filenames
        _extra_files = set()
        
        if h3_exclude_file:
            _extra_files = _load_exact_filenames(h3_exclude_file)
        
        all_files = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.endswith(".pt")
        ]
        if not all_files:
            raise RuntimeError(f"No .pt files found in {root_dir}")

        # Filter only by structure/antigen requirements, NOT by length
        filtered = []
        cached_lengths = []
        clono_keys = []
        epitope_keys = []

        length_stats = {
            "total": 0,
            "exceeds_max": 0,
            "requires_truncation": 0,
            "kept": 0
        }
        
        excluded_by_key = 0
        excluded_by_pdb = 0
        excluded_by_filename = 0
        
        for path in all_files:
            fname = os.path.basename(path).lower()
            _fkey, _pdb = _file_key_from_path(path)
        
            if fname in _extra_files:
                excluded_by_filename += 1
                continue
        
            if _fkey in _full_keys:
                excluded_by_key += 1
                continue
        
            if _pdb in _pdb_only:
                excluded_by_pdb += 1
                continue
            
            try:
                s = torch.load(path, map_location="cpu", weights_only=False)
                
                # Track length statistics
                heavy_seq = s["heavy"]["sequence_aa"]
                light_seq = s["light"]["sequence_aa"] or ""

                total_len = len(heavy_seq) + len(light_seq)
                effective_len = min(total_len, self.max_seq_len)

                length_stats["total"] += 1

                if total_len > self.max_seq_len:
                    length_stats["exceeds_max"] += 1
                    length_stats["requires_truncation"] += 1
                
                # Only filter by structure/antigen presence, not length
                if require_structure and s.get("coords") is None:
                    continue
                if require_antigen and s.get("antigen_coords") is None:
                    continue
                
                filtered.append(path)
                cached_lengths.append(effective_len)
                clono_keys.append(_extract_h3_key(heavy_seq, s.get("cdr_mask"), len(clono_keys)))
                # Epitope key: antigen cluster ID > target string > PDB ID fallback.
                # Cached here (zero extra I/O) so _sabdab_epitope_split never
                # needs to torch.load files again.
                _ekey = (
                    s.get("antigen_cluster_id")
                    or s.get("target")
                    or s.get("pdb_id", "")[:4].lower()
                    or f"__unk_{len(filtered)}"
                )
                epitope_keys.append(str(_ekey).lower().strip())
                length_stats["kept"] += 1    
                
            except Exception:
                continue
        print(
            f"[SAbDabDataset] Excluded "
            f"{excluded_by_filename} by filename, "
            f"{excluded_by_key} by key, "
            f"{excluded_by_pdb} by PDB"
        )
        
        self.files = filtered
        self.lengths = cached_lengths
        self.clono_keys = clono_keys
        self.epitope_keys = epitope_keys
        
        print(f"[SAbDabDataset] {root_dir}: {len(self.files)} files (max_seq_len={self.max_seq_len})")
        print(f"  Total samples: {length_stats['total']}")
        print(f"  Exceeds max_len: {length_stats['exceeds_max']} "
              f"({100*length_stats['exceeds_max']/max(1, length_stats['total']):.1f}%)")
        print(f"  Will truncate with CDR preservation: {length_stats['requires_truncation']}")
        print(f"  Kept after filtering: {length_stats['kept']}")
        
        # Length-based bucketing for efficient batching
        bucket_size = 50

        pairs = list(zip(self.files, self.lengths, self.clono_keys, self.epitope_keys))
        pairs.sort(key=lambda x: x[1])

        bucketed = []
        for i in range(0, len(pairs), bucket_size):
            bucket = pairs[i:i + bucket_size]
            random.shuffle(bucket)
            bucketed.extend(bucket)

        self.files       = [p for p, _, _, _ in bucketed]
        self.lengths     = [l for _, l, _, _ in bucketed]
        self.clono_keys  = [k for _, _, k, _ in bucketed]
        self.epitope_keys = [e for _, _, _, e in bucketed]

        print(f"[SAbDabDataset] Length bucketing complete (bucket_size={bucket_size})")

    def _smart_truncate(self, heavy_seq, light_seq, cdr_mask, coords, coord_mask,
                       interface_mask, antigen_coords, antigen_mask):
        """
        Smart truncation that preserves CDR regions when total length > max_seq_len.
        
        Strategy:
        1. If total_len <= max_seq_len: no truncation
        2. If total_len > max_seq_len:
           a. Keep ALL CDR positions intact (most important for binding)
           b. Sample framework positions uniformly to fill remaining space
           c. Maintain alignment between sequence and structural masks
        
        Returns:
            Tuple of (heavy_seq, light_seq, cdr_mask, coords, coord_mask,
                     interface_mask, antigen_coords, antigen_mask)
        """
        heavy_len = len(heavy_seq)
        light_len = len(light_seq)
        total_len = heavy_len + light_len
        
        # No truncation needed
        if total_len <= self.max_seq_len:
            return (heavy_seq, light_seq, cdr_mask, coords, coord_mask,
                   interface_mask, antigen_coords, antigen_mask)
        
        # Need to truncate - build combined sequence
        full_seq = heavy_seq + light_seq
        
        # Check if we have CDR mask for smart truncation
        has_cdr_mask = (cdr_mask is not None and len(cdr_mask) == total_len)
        
        if not has_cdr_mask:
            # Fallback: simple truncation (keep first max_seq_len positions)
            # Prioritize heavy chain (contains CDR3, the most important region)
            heavy_keep = min(heavy_len, int(self.max_seq_len * 0.65))
            light_keep = self.max_seq_len - heavy_keep
            
            return (
                heavy_seq[:heavy_keep],
                light_seq[:light_keep] if light_seq else "",
                cdr_mask[:self.max_seq_len] if cdr_mask is not None else None,
                coords[:self.max_seq_len] if coords is not None else None,
                coord_mask[:self.max_seq_len] if coord_mask is not None else None,
                interface_mask[:self.max_seq_len] if interface_mask is not None else None,
                antigen_coords,
                antigen_mask
            )
        
        # Smart truncation with CDR preservation
        cdr_positions = (cdr_mask > 0).nonzero(as_tuple=True)[0].tolist()
        framework_positions = (cdr_mask == 0).nonzero(as_tuple=True)[0].tolist()
        
        num_cdr = len(cdr_positions)
        num_framework = len(framework_positions)
        
        # Calculate framework positions to keep
        framework_to_keep = self.max_seq_len - num_cdr
        
        if framework_to_keep <= 0:
            # Edge case: CDRs alone exceed max_seq_len (very rare)
            # Keep first max_seq_len positions
            keep_indices = list(range(self.max_seq_len))
        elif framework_to_keep >= num_framework:
            # Keep all framework + all CDRs
            keep_indices = sorted(cdr_positions + framework_positions)
        else:
            # Sample framework positions uniformly
            step = num_framework / framework_to_keep
            sampled_fw = [framework_positions[min(int(i * step), num_framework-1)] 
                            for i in range(framework_to_keep)]
            keep_indices = sorted(cdr_positions + sampled_fw)
        
        # Ensure exactly max_seq_len positions
        keep_indices = keep_indices[:self.max_seq_len]
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
        
        # Build truncated sequence
        truncated_seq = "".join([full_seq[i] for i in keep_indices])
        
        # Split back into heavy/light
        heavy_kept = sum(1 for i in keep_indices if i < heavy_len)
        new_heavy = truncated_seq[:heavy_kept]
        new_light = truncated_seq[heavy_kept:]
        
        # Truncate all masks and coords using same indices
        new_cdr_mask = cdr_mask[keep_tensor] if cdr_mask is not None else None
        new_coords = coords[keep_tensor] if coords is not None else None
        new_coord_mask = coord_mask[keep_tensor] if coord_mask is not None else None
        new_interface_mask = interface_mask[keep_tensor] if interface_mask is not None else None
        
        return (new_heavy, new_light, new_cdr_mask, new_coords,
               new_coord_mask, new_interface_mask, antigen_coords, antigen_mask)

    def __len__(self) -> int:
        return len(self.files)

    def _tokenize(self, seq: str) -> torch.Tensor:
        return torch.tensor(
            [STOI.get(aa, STOI["<unk>"]) for aa in seq],
            dtype=torch.long
        )

    def __getitem__(self, idx: int) -> dict:
        sample = torch.load(self.files[idx], map_location="cpu",
                            weights_only=False)

        #  Load raw data 
        heavy_seq = sample["heavy"]["sequence_aa"]
        light_seq = sample["light"]["sequence_aa"] or ""

        coords = sample.get("coords")
        coord_mask = sample.get("coord_mask")
        antigen_coords = sample.get("antigen_coords")
        antigen_mask = sample.get("antigen_mask")
        antigen_seq = sample.get("antigen_seq", "")
        
        if antigen_seq:
            h = sample["heavy"]["sequence_aa"] or ""
            l = sample["light"]["sequence_aa"] or ""

            # Strong contamination checks only.
            # These usually indicate preprocessing / concatenation artifacts.
            ab_in_ag = (h and h in antigen_seq) or (l and l in antigen_seq)

            ag_in_ab = (
                antigen_seq and (
                    (h and antigen_seq in h) or
                    (l and antigen_seq in l)
                )
            )

            if ab_in_ag or ag_in_ab:
                antigen_seq    = ""
                antigen_coords = None
                antigen_mask   = None
        
        antigen_seq_tokens = (
            self._tokenize(antigen_seq) if antigen_seq else None
        )
        cdr_mask = sample.get("cdr_mask")
        interface_mask = sample.get("interface_mask")

        #  Smart truncation (CDR-preserving) 
        (heavy_seq, light_seq, cdr_mask, coords, coord_mask,
         interface_mask, antigen_coords, antigen_mask) = self._smart_truncate(
            heavy_seq, light_seq, cdr_mask, coords, coord_mask,
            interface_mask, antigen_coords, antigen_mask
        )

        #  Build final sequence 
        full_seq = heavy_seq + light_seq
        seq_tokens = self._tokenize(full_seq)
        seq_mask = torch.ones(len(seq_tokens), dtype=torch.bool)
        L = len(full_seq)

        #  Process coordinates 
        if coords is not None:
            coords = coords[:L]
            coord_mask = (coord_mask[:L] if coord_mask is not None 
                         else torch.ones(L, dtype=torch.bool))
            # Unsqueeze to [L, 1, 3] for multi-atom format compatibility
            coords = coords.unsqueeze(1)
            coord_mask = coord_mask.unsqueeze(1)

        #  Process CDR mask 
        if cdr_mask is not None:
            cdr_mask = cdr_mask[:L]

        #  Process interface mask 
        if interface_mask is not None:
            interface_mask = interface_mask[:L]

        #  Process antigen 
        if antigen_coords is not None:
            # Unsqueeze to [Lag, 1, 3] for consistency
            antigen_coords = antigen_coords.unsqueeze(1)
            if antigen_mask is not None:
                antigen_mask = antigen_mask.unsqueeze(1)

        #  Per-sample antigen-contact mask (CDR3-restricted, ≤8Å Cα) 
        # # Computed here (post-truncation) so indices always match the final
        # coords/cdr_mask actually returned - collate only pads, never
        # recomputes geometry. H3-ONLY (no L3), matching
        # path_isolation_by_contact()'s h3_b = cdr3_b & (chain == 0) bucket
        # definition exactly - L3 is intentionally never measured here.
        contact_mask = None
        if coords is not None and antigen_coords is not None and cdr_mask is not None:
            h3_mask_local = _extract_h3_mask(heavy_seq, cdr_mask)
            contact_mask = _compute_contact_mask(
                coords, coord_mask, h3_mask_local, antigen_coords, antigen_mask
            )

        #  Task mask 
        task_mask = {
            "AR": True,
            "MLM": False,
            "GEO": coords is not None,
            "IF": antigen_coords is not None,
        }

        # Interface label (sequence-level) 
        # Always 1.0 for (binding confirmed if antigen coords present), else None
        interface_labels = torch.tensor(1.0, dtype=torch.float32)

        return {
            # Metadata
            "source": sample.get("source", "SAbDab"),
            "source_id": 2 if sample.get("paired", False) else (
                        1 if (not heavy_seq and light_seq) else 0
                    ),
            "pdb_id": sample.get("pdb_id", ""),
            "dataset": "sabdab",
            "paired": sample.get("paired", False),
            "therapeutic": sample.get("therapeutic", False),
            "target": sample.get("target"),
            # Sequence
            "seq_tokens": seq_tokens,
            "seq_mask": seq_mask,
            "heavy_seq": heavy_seq,
            "heavy_len": len(heavy_seq),
            "heavy_len_actual": len(heavy_seq),
            "light_seq": light_seq,
            "light_len": len(light_seq),
            # Structure
            "coords": coords,
            "coord_mask": coord_mask,
            "antigen_coords": antigen_coords,
            "antigen_coord_mask": antigen_mask,
            "antigen_seq_tokens":  antigen_seq_tokens,
            # Masks
            "cdr_mask": cdr_mask,
            "cdr3_len_actual": sample.get("cdr3_len_actual", 0),
            "contact_mask": contact_mask,
            "interface_mask": interface_mask,
            "interface_labels": interface_labels,
            "has_residue_interface": interface_mask is not None,
            # Task
            "task_mask": task_mask,
        }



