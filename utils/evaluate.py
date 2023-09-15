import torch

def evaluate(model: torch.nn.Module, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        avg_loss (float): Average loss over the dataset.
        accuracy (float): Token-level accuracy (ignoring padding tokens).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass: outputs shape (batch_size, seq_len, vocab_size)
            outputs = model(batch_inputs)
            # Reshape for loss computation
            outputs = outputs.reshape(-1, outputs.size(-1))
            batch_targets = batch_targets.reshape(-1)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_inputs.size(0)

            # Compute predictions
            predictions = outputs.argmax(dim=-1)
            # Create mask to ignore padding tokens (token value 0)
            mask = batch_targets != 0
            correct_tokens += (predictions[mask] == batch_targets[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

