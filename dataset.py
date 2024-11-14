import os
import random

import torch
from torch.utils.data import Dataset

class GurkDataset(Dataset):

    def __init__(self, 
                 data_dir: str,
                 tokenizer, 
                 train=True, 
                 mask_p=0.15, 
                 max_len=128, 
                 p_only_masked=0.8
                 ):
        super().__init__()
        self.root = data_dir
        self.ids = self._get_ids(data_dir=data_dir)
        self.is_train = train
        self.mask_p = mask_p # Percentage of tokens that are masked.
        self.p_only_masked = p_only_masked # Of mask_p, how many are replaced by simple MASK token
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_ids(self, data_dir):
        return os.listdir(data_dir)

    def _read_file(self, path):
        with open(path, encoding="utf-8") as file:
            content = file.read().strip()
        return content.split()
    
    def _create_masks(self, input_ids):
        padding_mask = input_ids == self.tokenizer.pad_token_id
        # If not training, then no need to create lm mask
        if self.is_train is False:
            return padding_mask
        rand = torch.rand(input_ids.shape) # random probabilities for each token
        p_random = (1 - self.p_only_masked) # percentage of masked tokens that should be replaced with random tokens
        p_unchanged = p_random / 2 # Percentage of masked tokens that are left unchanged
        lm_mask = (rand <= self.mask_p) & (~ padding_mask) # Mask tokens with certain prob but not padding tokens.
        rand[~lm_mask] = torch.inf # Don't consider indices where lm_mask is False
        rand = rand / self.mask_p # Rescale according to masking percentage
        mask_random = (rand <= p_random) & (rand > p_unchanged)
        mask_unchanged = rand <= p_unchanged

        return lm_mask, padding_mask, mask_random, mask_unchanged

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        file_id = self.ids[index]
        path = os.path.join(self.root, file_id)
        content = self._read_file(path)
        # Tokenize with huggingface tokenizer.
        tok_output = self.tokenizer(
            content,
            is_split_into_words=True,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
            )
        input_ids = tok_output["input_ids"].squeeze(0)
        mask_out = self._create_masks(input_ids)
        if self.is_train:
            lm_mask, padding_mask, mask_random, mask_unchanged = mask_out
            masked = input_ids.detach().clone()
            masked[lm_mask] = self.tokenizer.mask_token_id
            masked[mask_random] = random.randrange(1, self.tokenizer.vocab_size) # Sample random tokens
            masked[mask_unchanged] = input_ids[mask_unchanged]
            return {
                "original_sequence": input_ids,
                "masked_sequence": masked,
                "padding_mask": padding_mask,
                "lm_mask": lm_mask
            }
        
        else:
            return {
                "original_sequence": input_ids,
                "padding_mask": mask_out,
            }
