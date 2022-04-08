def build_vocab(amino_acids: str) -> dict:
    """
    Build a vocabulary mapping amino acids to integer tokens.
    Reserve 0 for padding.
    """
    return {aa: i+1 for i, aa in enumerate(amino_acids)}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def tokenize_sequence(seq: str, vocab: dict) -> list:
    """
    Convert a protein sequence into a list of integer tokens.
    Unknown amino acids are mapped to 0.
    """
    return [vocab.get(aa, 0) for aa in seq]

