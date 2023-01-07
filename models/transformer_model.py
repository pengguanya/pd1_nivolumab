import math
import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding

class ProteinTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 128, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(emb_size, vocab_size)
        self.emb_size = emb_size

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len)
            src_mask: Optional mask for transformer.
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Transpose for transformer: (seq_len, batch_size)
        src = src.transpose(0, 1)
        src = self.embedding(src) * math.sqrt(self.emb_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output.transpose(0, 1)

