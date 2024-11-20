# read in data
from modules import FullModel
from dataset import GurkDataset
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from trainer import Trainer

torch.autograd.anomaly_mode.set_detect_anomaly(True)

train = "toy_data/train.txt"
max_len = 10
dim = 128
n_epochs = 50 # Will probably not lead to good results with the large vocab lol...

# Use a pretrained tokenizer, here BERT maybe something else??
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", local_files_only=True)
vocab_size = tokenizer.vocab_size


trainer = Trainer(
    tokenizer=tokenizer,
    train_dir="toy_data/train",
    test_dir="toy_data/test",
    model_params={
        "model_dim": 128,
        "num_heads": 4,
        "n_layers": 6, 
        "dropout": 0.01
    },
    optim_params={
        "lr": 0.1
    }, 
    optimizer=torch.optim.SGD,
    n_epochs=50,
    batch_size=2,
    mask_p=0.15,
    max_len=32
)

trainer.train(
    out_dir="checkpoints",
    start_from_chp=False,
    save_steps=5,
    chp_path="checkpoints\checkpoint-5-1.pt"
)

# Saving model
# Evaluation
test_data = trainer._get_test_dataloader()
trainer.model.eval()

# Pretty ugly evaluation for one test case, needed to make this more efficient, prettier later...
for test in test_data:
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
        pred = trainer.model(input_ids, key_padding_mask=pad_mask, pred_mask=mask_eye)
        pred = pred["masked_preds"]
        pred = pred.argmax(dim=1)
    # Prediction example
    print(f"True Sequence: {tokenizer.decode(y_true)}")
    print(f"Prediction: {tokenizer.decode(pred)}")
    correct = sum(pred == y_true)
    print(f"Accuracy {correct/y_true.shape[0]}")
