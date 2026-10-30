# data/loaders/oas_loader.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class OASDataset(Dataset):
    """
    Fast OAS dataset reading from pre-split train/ or val/ subdirectory.
    Sequential index access = NFS-friendly, no random seeks during training.
    Falls back to legacy single-dir format if no split subdirs found.
    """
    def __init__(self, path: str, split: str = "train"):
        """
        path:  root mmap directory (contains train/ and val/ subdirs)
        split: "train" or "val"
        """
        split_path = os.path.join(path, split)
        if os.path.exists(os.path.join(split_path, "meta.json")):
            self.path       = split_path
            self._pre_split = True
        else:
            # Legacy fallback: old mmap without pre-split dirs
            self.path       = path
            self._pre_split = False

        meta_path = os.path.join(self.path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {self.path}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.total        = self.meta["total_samples"]
        self.max_len      = self.meta["max_len"]
        self.pad_id       = self.meta.get("pad_id", 0)
        self.has_full_cdr = self.meta.get("has_full_cdr_masks", False)

        source_map         = self.meta.get("source_map", {"heavy": 0, "light": 1, "paired": 2})
        self._source_names = {v: k for k, v in source_map.items()}

        p = self.path

        # Core arrays - always present
        self._tokens = np.memmap(
            os.path.join(p, "tokens.bin"),
            dtype=np.int16, mode="r", shape=(self.total, self.max_len)
        )
        self._lengths = np.memmap(
            os.path.join(p, "lengths.bin"),
            dtype=np.int16, mode="r", shape=(self.total,)
        )

        # CDR masks - full mask always present, individual masks optional
        if self.has_full_cdr:
            self._cdr_full = np.memmap(
                os.path.join(p, "cdr_full_mask.bin"),
                dtype=np.uint8, mode="r", shape=(self.total, self.max_len)
            )
            print(f"[OASDataset] Loaded with FULL CDR masks (CDR1+2+3)")

            # Individual CDR masks - optional, only load if present
            cdr1_path = os.path.join(p, "cdr1_mask.bin")
            cdr2_path = os.path.join(p, "cdr2_mask.bin")
            cdr3_path = os.path.join(p, "cdr3_mask.bin")

            if os.path.exists(cdr1_path):
                self._cdr1 = np.memmap(
                    cdr1_path, dtype=np.uint8, mode="r",
                    shape=(self.total, self.max_len)
                )
                self._cdr2 = np.memmap(
                    cdr2_path, dtype=np.uint8, mode="r",
                    shape=(self.total, self.max_len)
                )
                self._cdr3 = np.memmap(
                    cdr3_path, dtype=np.uint8, mode="r",
                    shape=(self.total, self.max_len)
                )
                print(f"[OASDataset] Individual CDR1/2/3 masks loaded")
            else:
                self._cdr1 = self._cdr2 = self._cdr3 = None
                print(f"[OASDataset] Individual CDR masks not found - using full mask only (expected)")
        else:
            # Old format - cdr_mask.bin is CDR3 only
            self._cdr_full = np.memmap(
                os.path.join(p, "cdr_mask.bin"),
                dtype=np.uint8, mode="r", shape=(self.total, self.max_len)
            )
            self._cdr1 = self._cdr2 = self._cdr3 = None
            print(f"[OASDataset] Loaded with CDR3-only masks (legacy format)")

        # Sources
        sources_path = os.path.join(p, "sources.bin")
        if os.path.exists(sources_path):
            self._sources = np.memmap(
                sources_path, dtype=np.uint8, mode="r", shape=(self.total,)
            )
        else:
            self._sources = None
            print(f"[OASDataset] WARNING: sources.bin not found")

        # Clonotype hash
        clono_path = os.path.join(p, "clono_hash.bin")
        if os.path.exists(clono_path):
            self._clono_hashes = np.memmap(
                clono_path, dtype=np.uint32, mode="r", shape=(self.total,)
            )
            print(f"[OASDataset] clono_hash.bin loaded")
        else:
            self._clono_hashes = None
            print(f"[OASDataset] clono_hash.bin absent - clonotype info unavailable")

        # CDR3 length
        cdr3len_path = os.path.join(p, "cdr3_len.bin")
        if os.path.exists(cdr3len_path):
            self._cdr3_lens = np.memmap(
                cdr3len_path, dtype=np.uint8, mode="r", shape=(self.total,)
            )
            print(f"[OASDataset] cdr3_len.bin loaded")
        else:
            self._cdr3_lens = None
            print(f"[OASDataset] cdr3_len.bin absent - CDR3 length embedding disabled")
            
        # Heavy-chain length for paired sequences (needed for DualChainHead chain_mask)
        heavy_len_path = os.path.join(p, "heavy_len.bin")
        if os.path.exists(heavy_len_path):
            self._heavy_lens = np.memmap(
                heavy_len_path, dtype=np.uint16, mode="r", shape=(self.total,)
            )
            print(f"[OASDataset] heavy_len.bin loaded")
        else:
            self._heavy_lens = None
            print(f"[OASDataset] heavy_len.bin absent - deriving from CDR blocks for paired")

        print(
            f"[OASDataset] split={split!r} | "
            f"{'pre-split' if self._pre_split else 'legacy single-dir'} | "
            f"{self.total:,} samples | {self.path}"
        )
        
    def _derive_heavy_len_from_cdr(self, idx: int, length: int) -> int:
        """
        Approximate heavy-chain length for paired sequences by finding
        the start of the 4th CDR block (CDR1L) in the combined CDR mask.

        For OAS paired sequences, CDR blocks are:
          [0]=CDR1H  [1]=CDR2H  [2]=CDR3H  [3]=CDR1L  [4]=CDR2L  [5]=CDR3L
        CDR1L starts at exactly heavy_len (since light CDR masks are offset
        by len(h_s) during mmap construction in 2_convert_oas_mmap.py).

        Returns length//2 as a fallback if block structure is unexpected.
        """
        import numpy as np
        cdr_np = self._cdr_full[idx, :length].astype(np.int8)
        padded = np.concatenate(([0], cdr_np, [0]))
        diff   = np.diff(padded)
        starts = np.where(diff == 1)[0]
        if len(starts) >= 4:
            return int(starts[3])   # start of 4th block = start of CDR1L = heavy_len
        return length // 2           # crude fallback

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        length = int(self._lengths[idx])
        if length == 0:
            return None

        tokens_np  = self._tokens[idx, :length].astype(np.int64)
        seq_tokens = torch.from_numpy(tokens_np.copy())
        seq_mask   = torch.ones(length, dtype=torch.bool)

        # Full CDR mask (CDR1+2+3 combined) - always present
        cdr_np   = self._cdr_full[idx, :length].astype(bool)
        cdr_mask = torch.from_numpy(cdr_np.copy())

        # Individual CDR masks - optional, only if files exist
        extra = {}
        if self._cdr1 is not None:
            extra["cdr1_mask"] = torch.from_numpy(
                self._cdr1[idx, :length].astype(bool).copy()
            )
            extra["cdr2_mask"] = torch.from_numpy(
                self._cdr2[idx, :length].astype(bool).copy()
            )
            extra["cdr3_mask"] = torch.from_numpy(
                self._cdr3[idx, :length].astype(bool).copy()
            )

        # Source info
        if self._sources is not None:
            src_id     = int(self._sources[idx])
            oas_source = self._source_names.get(src_id, "unknown")
            paired     = (src_id == 2)
            source_id  = src_id
        else:
            src_id     = 0
            oas_source = "unknown"
            paired     = False
            source_id  = 0

        clono_hash = int(self._clono_hashes[idx]) if self._clono_hashes is not None else 0
        cdr3_len   = int(self._cdr3_lens[idx])    if self._cdr3_lens    is not None else 0

        # heavy_len_actual: needed by build_chain_position_mask for paired sequences
        if self._heavy_lens is not None:
            heavy_len_actual = int(self._heavy_lens[idx])
        elif src_id == 2:   # paired - derive from CDR block structure
            heavy_len_actual = self._derive_heavy_len_from_cdr(idx, length)
        else:
            heavy_len_actual = None   # unpaired heavy/light - not needed
        
        
        return {
            "seq_tokens":         seq_tokens,
            "seq_mask":           seq_mask,
            "cdr_mask":           cdr_mask,   # full CDR1+2+3 mask
            **extra,                           # cdr1/2/3 masks if available
            "coords":             None,
            "coord_mask":         None,
            "antigen_coords":     None,
            "antigen_seq_tokens": None,
            "paired":             paired,
            "source":             oas_source,
            "source_id":          source_id,
            "species":            "human",
            "dataset":            "oas",
            "clono_hash":         clono_hash,
            "cdr3_len_actual":    cdr3_len,
            "heavy_len_actual":   heavy_len_actual,
            "task_mask": {
                "AR":  True,
                "MLM": True,
                "GEO": False,
                "IF":  False,
            },
        }
