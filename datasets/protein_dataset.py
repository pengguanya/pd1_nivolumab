import torch
from torch.utils.data import Dataset
from utils.tokenization import tokenize_sequence

class ProteinDataset(Dataset):
    """
    A dataset for protein sequences.
    For demonstration, this dataset replicates a single protein sequence.
    """
    def __init__(self, sequences, seq_length: int, vocab: dict):
        """
        Args:
            sequences: List of (id, sequence) tuples.
            seq_length: Desired sequence length (for padding/truncation).
            vocab: Dictionary mapping amino acids to tokens.
        """
        self.sequence = sequences[0][1]  # use the first sequence for demo
        self.seq_length = seq_length
        self.vocab = vocab

    def __len__(self):
        # For demo, replicate the sequence 1000 times.
        return 1000

    def __getitem__(self, idx):
        tokens = tokenize_sequence(self.sequence, self.vocab)
        # Pad or truncate sequence
        if len(tokens) < self.seq_length:
            tokens = tokens + [0] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens, tokens  # (input, target) for self-supervised learning

