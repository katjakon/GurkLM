import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPClassifier(torch.nn.Module):
    """Set up neural network for POS tagging.

    This network consists of two layers: one RoBERTa embedding layer and on top
    of that a linear layer for classification. The linear layer projects to the
    final set of classes.

    Args:
        num_classes (int): the number of classes
    """
    def __init__(self, num_classes, model, dim):
        super(MLPClassifier, self).__init__()
        self.model = model
        self.disable_grad(self.model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, num_classes),
        )
    def disable_grad(self, module):
      for param in module.parameters():
        param.requires_grad = False

    def count_trainable_params(self):
      sum_el = 0
      for _, param in self.named_parameters():
        if param.requires_grad is True:
          sum_el += torch.numel(param)
      return sum_el

    def forward(self, x, padding_mask):
        # Retrieve embedding representation first
        self.model.eval()
        with torch.no_grad():
          emb = self.model(token_ids=x,key_padding_mask=padding_mask, pred_mask=None)["representations"]
        # Feed through mlp.
        return self.mlp(emb)

def accuracy(gold, pred, ignore_index=-100):
    """Calculate the accuracy for a given set of predictions."""
    # List containing 1 for each correct prediction, 0 for each incorrect one
    filtered_preds = [int(g == p) for g, p in zip(gold, pred) if g != -100]
    # (correct, total)
    return sum(filtered_preds), len(filtered_preds)

def get_pred_metrics(model, batch, device, loss_fn, pad_token_id=0):
    """Get all necessary metrics for one forward pass of a batch.

    These include: batch loss, number of correct predictions and total count
    of items in this batch
    """
    inputs = batch["input_ids"].to(device)
    pad_mask  = inputs == pad_token_id
    # Get prediction
    pred = model(inputs, pad_mask)
    pred = pred.flatten(end_dim=1)
    # Get gold labels
    gold_labels = batch["labels"].to(device)
    gold_labels = gold_labels.flatten(end_dim=1)
    # Calculate loss
    batch_loss = loss_fn(pred, gold_labels)
    # Calculate accuracy, get loss
    pred = torch.argmax(pred, dim=1)
    batch_correct, batch_total = accuracy(gold_labels, pred)
    return batch_loss, batch_correct, batch_total

def train_model(model, dataloader_trn, dataloader_val, lr, n_epochs, pad_token_id=0):
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
    )
  loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
  logs = []
  # Training loop
  for epoch in range(n_epochs):
    # Set model into training mode
    model.train()
    # Initialize counter
    train_total = 0
    train_correct = 0
    train_loss = 0
      # Work through batches
    for batch in tqdm(dataloader_trn, desc=f"Epoch {epoch}"):
      # Feed forward, get loss and counts for accuracy calculation
      batch_loss, batch_correct, batch_total = get_pred_metrics(model, batch, device, loss_fn=loss_function, pad_token_id=pad_token_id)
      train_total += batch_total
      train_correct += batch_correct
      train_loss += batch_loss.item()
      # Update step
      optimizer.zero_grad()
      batch_loss.backward()
      optimizer.step()
      train_accuracy = train_correct / train_total
          # Testing on validation set
    with torch.no_grad():
      # Set model into evaldation mode
      model.eval()
      # Initialiize counter
      val_total = 0
      val_correct = 0
      val_loss = 0
      for batch in tqdm(dataloader_val, desc=f"Validating..."):
        # Get loss and counts for accuracy calculation
        batch_loss, batch_correct, batch_total = get_pred_metrics(model, batch, device, loss_function)
        val_loss += batch_loss.item()
        val_total += batch_total
        val_correct += batch_correct
        val_accuracy = val_correct / val_total
    print(f"Train Loss: {train_loss}\t Train Accuracy {train_accuracy}")
    print(f"Val Loss: {val_loss}\t Val Accuracy {val_accuracy}")
    # Update plot
    epoch_results = {"Loss": train_loss,
                    "Accuracy": train_accuracy,
                    "val_Loss": val_loss,
                    "val_Accuracy": val_accuracy}
    logs.append(epoch_results)
  return logs

