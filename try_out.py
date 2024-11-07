# read in data
from toy_vocab import VOCAB
from modules import FullModel, Embedding
import torch 
import torch.nn as nn

torch.autograd.anomaly_mode.set_detect_anomaly(True)
train = "toy_data/train.txt"
max_len = 16
dim = 32

# Read in data
train_data = []
with open(train) as train_f:
    for line in train_f:
        train_data.append(line.split())

# ========= Should be handled by tokenizer later ==========
# Map to token id
vocab_dict = {word: idx for idx, word in enumerate(VOCAB)}
vocab_size = len(vocab_dict)

train_ids = []
for x in train_data:
    ids = []
    for token in x:
        ids.append(vocab_dict[token])
    # Padding!
    len_seq = len(ids)
    if len_seq < max_len:
        pad_len = max_len - len_seq
        ids.extend([vocab_dict["[PAD]"]] * pad_len
            )
    train_ids.append(ids)
    ids = []

# Convert train ids to tensor
train_ids = torch.IntTensor(train_ids).type(torch.int64) # (n_sequences, sequence_length)
# ============================================================== 

att_pad_mask = train_ids == vocab_dict["[PAD]"]
att_pad_mask = att_pad_mask.T
print(att_pad_mask)
model = FullModel(
    model_dim=8, 
    vocab_size=vocab_size,
    num_heads=4,
    n_layers=2, 
    dropout=0.01
)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

y_true = nn.functional.one_hot(train_ids).type(torch.float64)
y_true = torch.flatten(y_true, end_dim=1)
y_true = y_true.argmax(dim=1)
print(y_true)

for epoch in range(50):

    print(f"EPOCH {epoch}")
    model.train()
    pred = model(train_ids, key_padding_mask=att_pad_mask)
    print(pred)
    pred = torch.flatten(pred, end_dim=1)
    print(pred.argmax(dim=1))
    loss = loss_function(pred, y_true)
    print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()

mask = y_true != 0
y_true = y_true[mask]
pred = pred.argmax(dim=1)[mask]
correct = sum(pred == y_true)
print(correct/y_true.shape[0])
