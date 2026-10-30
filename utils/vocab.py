# utils/vocab.py - FIXED VERSION
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD = "<pad>"
UNK = "<unk>"
START = "<s>"
END = "</s>"
MASK = "<mask>"  # Explicit mask token for MLM

def build_vocab():
    """Build vocabulary with explicit MASK token."""
    toks = [PAD, UNK, START, END, MASK] + list(AMINO_ACIDS)
    stoi = {s: i for i, s in enumerate(toks)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

STOI, ITOS = build_vocab()

def tokenize(seq, max_len=None):
    """Tokenize sequence, mapping unknown AAs to UNK."""
    tokens = [START] + list(seq) + [END]
    if max_len:
        tokens = tokens[:max_len]
        # ensure END exists
        if tokens[-1] != END:
            tokens[-1] = END
    ids = [STOI.get(t, STOI[UNK]) for t in tokens]
    return ids

def detokenize(ids):
    """Convert token IDs back to sequence, removing special tokens."""
    return "".join([ITOS[i] for i in ids if ITOS[i] not in (PAD, START, END, MASK, UNK)])