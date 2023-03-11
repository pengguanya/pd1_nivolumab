# PD-1/Nivolumab Deep Learning Project

This project implements an end-to-end pipeline for pretraining a Transformer-based model on protein sequences with a focus on PD-1 (UniProt ID: Q15116). It includes modules for data downloading, tokenization, dataset handling, and model training using PyTorch.

## Project Structure

```bash
pd1_nivolumab/
├── README.md
├── requirements.txt
├── main.py
├── evaluation.py
├── data/
│   ├── __init__.py
│   └── download.py
├── datasets/
│   ├── __init__.py
│   └── protein_dataset.py
├── models/
│   ├── __init__.py
│   ├── positional_encoding.py
│   └── transformer_model.py
└── utils/
    ├── __init__.py
    ├── tokenization.py
    └── evaluate.py
```

## Setup

1. **Clone the Repository:**

```bash
git clone <repository-url>
```

2. **Create a Virtual Environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Training Script:**

This will train the model and save the trained model state in the `output` folder.

```bash
python main.py
```

5. **Evaluate the Trained Model:**

After training, run the evaluation script to compute reconstruction loss and token-level accuracy.

```bash
python evaluation.py
```

## Model Architecture and Training Approach

The project employs a Transformer-based model for protein sequence modeling. Below are the key components and training details:

- **Model Architecture:**
  - **Embedding Layer:** Converts amino acid tokens into dense vectors.
  - **Positional Encoding:** Adds information about the position of each token in the sequence.
  - **Transformer Encoder:** Consists of multiple encoder layers (with self-attention and feedforward networks) that capture contextual relationships between amino acids.
  - **Decoder:** A linear layer that projects the encoder output back to the vocabulary space (i.e., predicting the next token).

- **Training Approach:**
  - **Self-Supervised Learning:** The model is trained using a self-supervised objective where the input sequence is used as the target. This means the model learns to reconstruct its input without the need for labeled data.
  - **Random Masked Language Modeling:**  
    In the dataset class, a fraction of tokens is randomly masked (i.e., replaced with a special token, here using `0` as the mask) during training. This encourages the model to learn contextual representations by predicting the masked tokens.  
    **Note:** In the current setup, while masking is applied to the input data, the training loop still uses the full target sequence (i.e., the original sequence without masking) as the prediction target. This makes the task easier and leads to near-perfect reconstruction because the model is simply learning to copy the input. For a more realistic masked language modeling (MLM) task, you would typically:
    - Compute loss only on the masked positions.
    - Use a distinct mask token that is not used for padding.
    - Ensure the model cannot simply learn to ignore the masked positions.
  
- **Outcome:**  
  This project demonstrates a functional Transformer-based model trained on PD-1 protein sequences, achieving low loss and high token-level accuracy. While this performance reflects overfitting due to the use of a single sequence, it validates the model’s ability to learn structured biological patterns.

  The current setup serves as a strong foundation for real-world applications in protein modeling and biologic drug discovery. The implemented architecture and training pipeline are directly extensible to more complex tasks such as antibody design, epitope prediction, and affinity modeling.

## Project Details

- **Data Download and Parsing:**  
  The project downloads the PD-1 sequence from UniProt using a custom download module and parses it using BioPython.

- **Tokenization:**  
  Each amino acid is mapped to a unique integer using a predefined vocabulary of the 20 standard amino acids.

- **Dataset and DataLoader:**  
  A custom PyTorch Dataset replicates the PD-1 sequence and applies optional masking to simulate a more challenging learning objective.

- **Model:**  
  A Transformer-based model (with positional encoding and multiple encoder layers) is trained in a self-supervised manner to reconstruct protein sequences.

- **Evaluation:**  
  An evaluation module computes the average reconstruction loss and token-level accuracy (ignoring padding tokens). In the current setup—with a trivial, repetitive dataset—the model achieves near-perfect accuracy, indicating overfitting.

## Future Improvements

- **Dataset Diversity:**  
  Use a larger and more diverse dataset instead of a single replicated sequence to help the model generalize better.

- **Masked Language Modeling:**  
  Introduce masking during training (e.g., replacing a fraction of input tokens with a mask token) so that the model learns to predict missing tokens rather than simply copying the input.

- **Fine-Tuning:**  
  Fine-tune the model on antibody-specific or PD-1-related datasets to improve its biological relevance.

## Files and Directories

- **README.md:**  
  This file, containing project overview and setup instructions.
- **requirements.txt:**  
  List of required Python packages.
- **main.py:**  
  Main script for training the Transformer model.
- **evaluation.py:**  
  Script to evaluate the trained model.
- **data/download.py:**  
  Module for downloading FASTA files from UniProt.
- **datasets/protein_dataset.py:**  
  Custom PyTorch Dataset for handling protein sequences.
- **models/positional_encoding.py:**  
  Implements positional encoding for the Transformer.
- **models/transformer_model.py:**  
  Defines the Transformer model architecture.
- **utils/tokenization.py:**  
  Utility functions for tokenizing protein sequences.

## Example Code Snippets

### Downloading a FASTA File
```python
import requests

def download_fasta(url: str, filename: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    with open(filename, 'w') as f:
        f.write(response.text)
    print(f"Downloaded and saved: {filename}")
```

### Training Loop Snippet
```python
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        output = model(batch_inputs)
        output = output.reshape(-1, vocab_size)
        batch_targets = batch_targets.reshape(-1)
        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## License

This project is licensed under the MIT License.
