# utils/mlm.py
import torch
from utils.vocab import STOI, PAD, START, END, UNK, MASK

PAD_ID = STOI[PAD]
START_ID = STOI[START]
END_ID = STOI[END]
UNK_ID = STOI[UNK]
MASK_ID = STOI[MASK]

def apply_mlm_mask(
    seq_tokens: torch.Tensor,
    seq_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    mask_prob: float = 0.15,
    cdr3_mask: torch.Tensor | None = None,
    cdr12_mask: torch.Tensor | None = None,
    debug: bool = False,
):
    """
    CDR-tiered MLM masking.

    When cdr3_mask / cdr12_mask are supplied, per-position masking
    probabilities follow the published schedule:
        CDR3  -> p = 0.65
        CDR1/2 -> p = 0.45
        Framework -> p = 0.08

    Otherwise falls back to uniform mask_prob over candidate_mask.
    """
    device = seq_tokens.device
    B, L   = seq_tokens.shape

    # Build candidate set
    candidate = seq_mask.clone()
    if candidate_mask is not None:
        candidate &= candidate_mask
    for tok in (PAD_ID, START_ID, END_ID, MASK_ID):
        candidate &= (seq_tokens != tok)

    if candidate.sum() == 0:
        labels      = torch.full_like(seq_tokens, -100)
        empty_mask  = torch.zeros_like(seq_tokens, dtype=torch.bool)
        if debug:
            print("[MLM] WARNING: No maskable positions found!")
        return seq_tokens, labels, empty_mask

    # ── Per-position masking probability ─────────────────────────────
    if cdr3_mask is not None and cdr12_mask is not None:
        p_map = torch.full((B, L), 0.08, device=device)          # framework
        p_map[cdr12_mask.bool()] = 0.45
        p_map[cdr3_mask.bool()]  = 0.65
        p_map = p_map * candidate.float()                         # zero non-candidates
    else:
        p_map = torch.full((B, L), mask_prob, device=device) * candidate.float()

    rand     = torch.rand(B, L, device=device)
    mlm_mask = rand < p_map

    # Guarantee ≥1 masked token per sample
    for b in range(B):
        if mlm_mask[b].sum() == 0:
            valid = candidate[b].nonzero(as_tuple=False)
            if len(valid) > 0:
                mlm_mask[b, valid[0].item()] = True

    # Labels
    labels = seq_tokens.clone()
    labels[~mlm_mask] = -100

    # 80 / 10 / 10 rule
    rand2      = torch.rand(B, L, device=device)
    seq_tokens = seq_tokens.clone()

    mask_80 = mlm_mask & (rand2 < 0.8)
    seq_tokens[mask_80] = MASK_ID

    random_mask = mlm_mask & (rand2 >= 0.8) & (rand2 < 0.9)
    if random_mask.any():
        random_tokens = torch.randint(5, len(STOI), seq_tokens.shape, device=device)
        seq_tokens[random_mask] = random_tokens[random_mask]

    return seq_tokens, labels, mlm_mask