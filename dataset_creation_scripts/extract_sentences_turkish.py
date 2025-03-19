import re

from datasets import load_dataset
from nltk.tokenize import sent_tokenize

from huggingface_hub import login


# Login
auth_token = None
login(token=auth_token)

# Turkish webcrawl
dataset = load_dataset("oscar-corpus/OSCAR-2301",
                        trust_remote_code=True,
                        language="tr",
                        streaming=True,
                        split="train")

# Set up log file
with open("data/log_tr_monolingual_new.txt", "w", encoding="utf-8") as file_out:
    file_out.write("Beginning data collection\n\n")

# All letters used in German and Turkish, and some punctuation characters
allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ßÄÖÜäöüçşğıâİÇŞĞ’!"«»#$%&\'()*+, -./:;<=>?@[]^_`{|}~‘„”“…–')

# List to collect sentences, counters for sentences and tokens
sents = []
sent_count = 0
token_count = 0

x = 10000  # Final sent number for current file
file_lim = 10000  # General file limit
token_req = 250000000  # Stop if we reach this many tokens

# Go through individual items in dataset
for d in dataset:
    text = d["text"]
    text = text.replace("\n", " ")
    # Filter out sentences that contain strange characters (encoding errors)
    if set(text).difference(allowed_chars):
        continue
    # Split into sentences
    s = sent_tokenize(text, language="turkish")
    # Filter out very long sentences and make sure all of them contain letters
    s = [i for i in s if len(i.split()) < 51 and re.search(r'[a-zA-Z]', i)]
    sent_count += len(s)
    sents += s
    # Time to save to file
    if sent_count >= x:
        # Save sentences to file
        sents_curr = sents[:x]
        with open(f"data/tr_monolingual_new/tr_monolingual_new_{x-file_lim}_{x}.txt", "w", encoding="utf-8") as file_out:
            for sent in sents_curr:
                token_count += len(sent.split())
                file_out.write(f"{sent}\n")
        # Write entry to log file
        with open("data/log_tr_monolingual_new.txt", "a", encoding="utf-8") as file_out:
            file_out.write(f"sent_count: {sent_count}\n")
            file_out.write(f"token_count: {token_count}\n")
            file_out.write(f"already finished {x} sentences, {token_count} tokens\n")
            file_out.write("-"*50 + "\n")
        # Continue with rest of sentences
        sents = sents[x:]
        x += file_lim  # New sent number for next file
        # If token requirement is reached: Stop and write final counts to log
        if token_count >= token_req:
            with open("data/log_tr_monolingual_new.txt", "a", encoding="utf-8") as file_out:
                file_out.write(f"finished with {token_count} tokens\n")
                file_out.write("-"*50 + "\n")
            break

# Tail of log file
with open("data/log_tr_monolingual_new.txt", "a", encoding="utf-8") as file_out:
    file_out.write(f"finished the loop\n")
    file_out.write("-"*50 + "\n")

