# read in data
from modules import FullModel
from dataset import GurkDataset
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

torch.autograd.anomaly_mode.set_detect_anomaly(True)

train = "toy_data/train.txt"
max_len = 10
dim = 128
n_epochs = 50 # Will probably not lead to good results with the large vocab lol...

# Use a pretrained tokenizer, here BERT maybe something else??
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", local_files_only=True)
vocab_size = tokenizer.vocab_size

# Gurk data set init, custom dataset
dataset = GurkDataset(
    data_dir="toy_data/train",
    tokenizer=tokenizer,
    train=True,
    max_len=max_len,
    mask_p=0.15
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=5,
    shuffle=True
)

# Model initialization
model = FullModel(
    model_dim=dim, 
    vocab_size=vocab_size,
    num_heads=2,
    n_layers=6, 
    dropout=0.05,
    max_len=max_len
)
# Define Loss criterion and optimizer
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    model.train()
    print(f"EPOCH {epoch}")
    for batch in dataloader:
        batch_input_ids = batch["masked_sequence"] # Here randomly some tokens have been replace by [MASK].
        batch_padding_mask = batch["padding_mask"] # Masks which specifies which tokens are [PAD]
        batch_lm_mask = batch["lm_mask"] # Masks which specifies where tokens have been replace by [MASK]
        batch_y_true = batch["original_sequence"] # True tokens without mask
        batch_y_true = torch.flatten(batch_y_true[batch_lm_mask])

        batch_pred = model(
            batch_input_ids,
            key_padding_mask=batch_padding_mask,
            pred_mask=torch.flatten(batch_lm_mask))

        loss = loss_function(batch_pred, batch_y_true)
        if batch_y_true.shape[0] != 0: # We might have masked no tokens by chance, avoid zero division
            batch_acc = sum(batch_pred.argmax(dim=1) == batch_y_true) / (batch_y_true.shape[0])
            print(f"Batch Accuracy: {batch_acc}")
        print(f"Batch Loss: {loss}")
        optim.zero_grad()
        loss.backward()
        optim.step()

# Evaluation
test_data = GurkDataset(
    data_dir="toy_data/test",
    tokenizer=tokenizer,
    train=False,
    max_len=max_len
)
dataloader = DataLoader(test_data)
model.eval()

# Pretty ugly evaluation for one test case, needed to make this more efficient, prettier later...
for test in dataloader:
    input_ids = test["original_sequence"]
    pad_mask = test["padding_mask"]
    # Mask each token exactly once
    input_ids = input_ids.repeat(input_ids.shape[1], 1)
    pad_mask = pad_mask.repeat(pad_mask.shape[1], 1)
    eye = torch.eye(input_ids.shape[1])
    mask_eye = (eye == 1.0) & ~ pad_mask # Don't Mask/predict padding tokens.
    input_ids[mask_eye] = tokenizer.mask_token_id
    y_true = test["original_sequence"].squeeze(1)
    y_true = y_true[y_true != tokenizer.pad_token_id]

    input_ids = input_ids[:y_true.shape[0], :]
    pad_mask = pad_mask[:y_true.shape[0], :]
    mask_eye = mask_eye[:y_true.shape[0], :]

    mask_eye = torch.flatten(mask_eye)
    with torch.no_grad():
        pred = model(input_ids, key_padding_mask=pad_mask, pred_mask=mask_eye)
        pred = pred.argmax(dim=1)
    # Prediction example
    print(f"True Sequence: {tokenizer.decode(y_true)}")
    print(f"Prediction: {tokenizer.decode(pred)}")
    correct = sum(pred == y_true)
    print(f"Accuracy {correct/y_true.shape[0]}")
