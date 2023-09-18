import os
import torch
from models.transformer_model import ProteinTransformer
from datasets.protein_dataset import ProteinDataset
from utils.tokenization import build_vocab, AMINO_ACIDS
from utils.evaluate import evaluate
from torch.utils.data import DataLoader
from Bio import SeqIO

# Build the vocabulary and set vocab size (same as during training)
vocab = build_vocab(AMINO_ACIDS)
vocab_size = len(vocab) + 1

# Instantiate the model with the same parameters used in training
model = ProteinTransformer(vocab_size=vocab_size, emb_size=128, nhead=8, num_layers=4,
                             dim_feedforward=256, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the output directory and load the saved model
output_dir = "output"
model_save_path = os.path.join(output_dir, "trained_model.pth")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
print("Trained model loaded for evaluation from", model_save_path)

def parse_fasta(filename: str):
    sequences = []
    with open(filename) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append((record.id, str(record.seq)))
    return sequences

pd1_sequences = parse_fasta("PD1_Q15116.fasta")
dataset = ProteinDataset(pd1_sequences, seq_length=128, vocab=vocab)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Evaluate the model
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
eval_loss, eval_accuracy = evaluate(model, dataloader, criterion, device)
print(f"Evaluation Loss: {eval_loss:.4f}, Token-level Accuracy: {eval_accuracy*100:.2f}%")
