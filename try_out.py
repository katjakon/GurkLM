# read in data
from toy_vocab import VOCAB
from modules import FullModel
import torch 
import torch.nn as nn

train = "toy_data/train.txt"
max_len = 16
dim = 8

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
model = FullModel(
    model_dim=8, 
    vocab_size=vocab_size,
    num_heads=1,
    n_layers=1, 
    dropout=0.01
)

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.001)

y_true = nn.functional.one_hot(train_ids).type(torch.float64)
print(y_true)
y_true_classes = y_true.argmax(dim=2)

for epoch in range(200):

    print(f"EPOCH {epoch}")
    model.train()
    pred = model(train_ids, key_padding_mask=None)
    loss = loss_function(pred, y_true)
    print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()

pred = model(train_ids, key_padding_mask=None)
pred = torch.argmax(pred, dim=2)
print(pred)
y_true = torch.argmax(y_true, dim=2)
correct = torch.sum(y_true == pred)
all_pred = torch.sum(torch.ones(y_true.shape))
print("Accuracy", correct/all_pred)
print(vocab_dict)
