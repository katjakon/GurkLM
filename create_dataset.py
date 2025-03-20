import argparse
import json
import os
# import random
# import re
# from typing import Dict, List, Optional

# from datasets import Dataset
# import torch
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from transformers import AutoTokenizer

from gurk.dataset import PackedCucumbers

# silence tqdm
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="Path to the directory containing the data files, each consisting lines of raw sentences")
parser.add_argument("out_dir", type=str, help="Directory to store the output_files at, will be created")
parser.add_argument("--seq-len", required=True, type=int, help="Length to which the sequences are to be packed")
parser.add_argument("--file-len", required=False, type=int, default=100000, help="How many lines to store in a file, defaults to 100000")
parser.add_argument("--tokenizer", required=False, type=str, default="bert-base-cased", help="Identifier of the tokenizer to use, defaults to 'bert-base-cased'")

args = parser.parse_args()

# Make sure all provided arguments are valid
assert os.path.isdir(args.data_dir), "Data directory not found"
os.mkdir(args.out_dir)  # Try to create directory for output (may not exist beforehand)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# Create dataset
gurks = PackedCucumbers(args.data_dir, tokenizer, max_seq_len=args.seq_len)
ds = gurks.create_giant_dataset()
out_dir = args.out_dir

# Save
file_len = args.file_len
c = 0  # Counter
lines = []
for instance in ds:
    # If file_len limit is reached: save to file
    if len(lines) == file_len:
        # Create file name
        file_name = f"file{c-len(lines)}_{c}.jsonlines"
        with open(os.path.join(out_dir, file_name), "w", encoding="utf-8") as f:
            for line in lines:
                json.dump(line, f)
                f.write("\n")
        lines = []  # Reset lines
    # Collect data items for current file
    instance["input_ids"] = instance["input_ids"].tolist()
    lines.append(instance)
    c += 1

# Save final lines as well
file_name = f"file{c-len(lines)}_{c}.jsonlines"
with open(os.path.join(out_dir, file_name), "w", encoding="utf-8") as f:
    for line in lines:
        json.dump(line, f)
        f.write("\n")

