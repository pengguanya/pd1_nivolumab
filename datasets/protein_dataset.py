import torch
from torch.utils.data import Dataset
from utils.tokenization import tokenize_sequence

class ProteinDataset(Dataset):
    """
    A dataset for protein sequences with an option for masked language modeling.
    """
    def __init__(self, sequences, seq_length: int, vocab: dict, mask_prob: float = 0.15):
        """
        Args:
            sequences: List of (id, sequence) tuples.
            seq_length: Desired sequence length (for padding/truncation).
            vocab: Dictionary mapping amino acids to tokens.
            mask_prob: Probability of masking each token.
        """
        self.sequence = sequences[0][1]  # For demonstration, using the first sequence.
        self.seq_length = seq_length
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.mask_token = 0  # We use 0 as both padding and mask for simplicity.
        
    def __len__(self):
        # For demo, replicate the sequence 1000 times.
        return 1000

    def __getitem__(self, idx):
        tokens = tokenize_sequence(self.sequence, self.vocab)
        # Pad or truncate the sequence
        if len(tokens) < self.seq_length:
            tokens = tokens + [0] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        # Create a copy for the target (the original tokens)
        target = tokens.clone()
        
        # For training, randomly mask tokens with mask_prob (ignoring padding)
        if self.mask_prob > 0:
            mask = (torch.rand(tokens.shape) < self.mask_prob) & (tokens != 0)
            # Set masked tokens to mask_token (here we use 0)
            tokens[mask] = self.mask_token

        return tokens, target

