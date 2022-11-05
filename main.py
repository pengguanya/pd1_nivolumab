import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.download import download_fasta
from datasets.protein_dataset import ProteinDataset
from models.transformer_model import ProteinTransformer
from utils.tokenization import tokenize_sequence, AMINO_ACIDS, build_vocab
from models.positional_encoding import PositionalEncoding  # (if you want to test separately)

# --------------------------
# Download and parse PD-1 sequence
# --------------------------
uniprot_url = "https://www.uniprot.org/uniprot/Q15116.fasta"
fasta_filename = "PD1_Q15116.fasta"
if not os.path.exists(fasta_filename):
    download_fasta(uniprot_url, fasta_filename)

# For simplicity, we read the FASTA file here using Biopython directly.
from Bio import SeqIO
def parse_fasta(filename: str):
    sequences = []
    with open(filename) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append((record.id, str(record.seq)))
    return sequences

pd1_sequences = parse_fasta(fasta_filename)
print("Parsed PD-1 sequence:")
print(pd1_sequences[0][0], pd1_sequences[0][1][:50] + "...")

# --------------------------
# Build Vocabulary and Tokenize Example Sequence
# --------------------------
vocab = build_vocab(AMINO_ACIDS)
print("Vocabulary:", vocab)

# Tokenize the PD-1 sequence (for demonstration)
tokenized_pd1 = tokenize_sequence(pd1_sequences[0][1], vocab)
print("Tokenized PD-1 sequence (first 50 tokens):", tokenized_pd1[:50])

# --------------------------
# Create Dataset and DataLoader
# --------------------------
# For demonstration, we use a fixed sequence length (e.g., 128)
dataset = ProteinDataset(pd1_sequences, seq_length=128, vocab=vocab)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------
# Instantiate Transformer Model
# --------------------------
vocab_size = len(vocab) + 1  # plus one for padding index 0
model = ProteinTransformer(vocab_size=vocab_size, emb_size=128, nhead=8, num_layers=4,
                             dim_feedforward=256, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------------------
# Training Setup
# --------------------------
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5

# --------------------------
# Training Loop
# --------------------------
model.train()
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        # Model output: (batch_size, seq_len, vocab_size)
        output = model(batch_inputs)
        # Reshape to (batch_size * seq_len, vocab_size)
        output = output.reshape(-1, vocab_size)
        batch_targets = batch_targets.reshape(-1)
        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

# --------------------------
# Sequence Generation Example
# --------------------------
def generate_sequence(model: nn.Module, start_seq: str, vocab: dict, max_len: int = 100, temperature: float = 1.0):
    model.eval()
    tokens = tokenize_sequence(start_seq, vocab)
    input_seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
    generated = tokens.copy()
    for _ in range(max_len - len(tokens)):
        with torch.no_grad():
            output = model(input_seq)  # (1, seq_len, vocab_size)
            next_token_logits = output[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_token)
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == 0:  # break if padding token is generated
                break
    # Build inverse vocabulary
    inv_vocab = {i: aa for aa, i in vocab.items()}
    generated_seq = "".join([inv_vocab.get(tok, "") for tok in generated if tok != 0])
    return generated_seq

generated_sequence = generate_sequence(model, start_seq="M", vocab=vocab, max_len=100)
print("Generated sequence:", generated_sequence)

