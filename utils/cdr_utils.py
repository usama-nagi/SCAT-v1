# utils/cdr_utils.py
# Derive CDR3 mask at runtime from cdr_full_mask.

import torch
import numpy as np

def extract_cdr3_mask(cdr_full_mask: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """
    Derive CDR3 positions from the combined CDR mask.

    Block layout after preprocessing:

      OAS layout - 3 blocks per chain:
        single chain : [CDR1, CDR2, CDR3]        -> CDR3 = block index 2
        paired       : [H1, H2, H3, L1, L2, L3]  -> CDR3 = blocks 2 and 5

      SAbDab v2 layout (IMGT-correct, 3 blocks per chain):
        single chain : [H1, H2, H3]               -> CDR3 = block index 2
        paired       : [H1, H2, H3, L1, L2, L3]  -> CDR3 = blocks 2 and 5

      SAbDab v1 legacy layout (Kabat-wrong, 1 block per chain):
        single chain : [CDR3-only]                -> CDR3 = block 0
        paired       : [H-CDR3, L-CDR3]           -> CDR3 = blocks 0 and 1

    The 6-block and 3-block cases are now identical between OAS and SAbDab v2.
    Legacy 1-block and 2-block cases still handled for backward compatibility.
    """
    B, L = cdr_full_mask.shape
    cdr3 = torch.zeros(B, L, dtype=torch.bool, device=cdr_full_mask.device)

    mask_np = cdr_full_mask.cpu().numpy().astype(np.uint8)

    for i in range(B):
        padded = np.concatenate(([0], mask_np[i], [0]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]

        n      = len(starts)
        src    = int(sources[i].item())
        paired = (src == 2)

        if n == 0:
            continue

        if paired:
            if n >= 6:
                cdr3[i, starts[2]:ends[2]] = True
                cdr3[i, starts[5]:ends[5]] = True
            elif n == 5:
                # H1H2H3L1L2 or H1H2H3L2L3 - take block 2 (H3) and block 4 (last L)
                cdr3[i, starts[2]:ends[2]] = True
                cdr3[i, starts[4]:ends[4]] = True
            elif n == 4:
                cdr3[i, starts[2]:ends[2]] = True
                cdr3[i, starts[3]:ends[3]] = True
            elif n == 3:
                # Paired but only heavy CDRs annotated - take block 2 (H3)
                cdr3[i, starts[2]:ends[2]] = True
            elif n == 2:
                # Legacy v1: 2 blocks = H3 + L3
                cdr3[i, starts[0]:ends[0]] = True
                cdr3[i, starts[1]:ends[1]] = True
            else:
                # n == 1
                cdr3[i, starts[0]:ends[0]] = True
        else:
            # Single chain
            if n >= 3:
                # OAS single chain OR SAbDab v2 single chain: 3 blocks
                # CDR3 = block 2 (last block = CDR3)
                cdr3[i, starts[2]:ends[2]] = True
            elif n == 2:
                # Partial single chain annotation: CDR2+CDR3 or CDR1+CDR3
                # Take the last block as CDR3
                cdr3[i, starts[-1]:ends[-1]] = True
            else:
                # n == 1: SAbDab v1 legacy single chain (CDR3 only)
                cdr3[i, starts[0]:ends[0]] = True

    return cdr3