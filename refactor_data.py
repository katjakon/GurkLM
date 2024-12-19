import os
import random
from transformers import AutoTokenizer

from typing import Dict, List, Optional

import torch
from torch.nn import functional as F

# from torch.utils.data import Dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm

# silence tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


class PackedCucumbers:

    def __init__(self, data_path, tokenizer, seed=13, max_seq_len=512, start_id=0, batch_num=0, shuffle=True):
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.start_batch = batch_num
        self.root = data_path
        self.files = {file_id: file_path for file_id, file_path in enumerate(os.listdir(data_path))}
        self.shuffeled_ids = list(self.files.keys())
        self.tokenizer = tokenizer
        if self.shuffle:
            random.seed(seed)
            random.shuffle(self.shuffeled_ids)
            print(self.shuffeled_ids)
        if start_id:
            self.shuffeled_ids = self.shuffeled_ids[self.shuffeled_ids.index(start_id):]
        self.batch_num = batch_num

    def _load(self, file_path):
        with open(file_path, encoding="utf-8") as file_in:
            lines = [{"text": line} for line in file_in]
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(lines)
            ds = Dataset.from_list(lines)
            ds = ds.map(self._tokenize)
        return PackedDataset(ds, max_seq_len=self.max_seq_len, split_across_pack=True)

    def _tokenize(self, instance):
        input_ids = self.tokenizer(instance["text"],
                                   return_attention_mask=False,
                                   add_special_tokens=False)["input_ids"]
        return {"input_ids": input_ids}


    def create_giant_dataloader(self, batch_size=32):
        file_paths = [os.path.join(self.root, self.files[i]) for i in self.shuffeled_ids]
        dataloader = DataLoader([i for f in file_paths for i in self._load(f).packs],
                                batch_size=batch_size)
        return dataloader

    def __len__(self):
        return len(self.ids)


class PackedDataset(Dataset):

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs: List[PACK_TYPE] = []
        self.previous_sample_boundary: int = 0
        self._pack()
        # self._create_attention_masks()

    def _create_attention_masks(self) -> None:
        for idx in range(len(self.packs)):
            i = self.packs[idx]
            tokenized_sentences = i["input_ids"]
            T = tokenized_sentences.size(0)
            eos_indices = torch.cumsum(i["seq_lens"], dim=0)
            reps = i["seq_lens"]
            # repeat each eos index n times along dimension 1 (n is the number of tokens in the sequence)
            repeated_idx = torch.repeat_interleave(eos_indices, reps).view(1,-1).expand(T, -1)
            # create tensor with all indices from 0 to T-1 repeated T times along dimesion 1
            mask_indices = torch.arange(T).view(-1,1).expand(-1, T)
            # create causal mask and additionally mask out all tokens from preceeding sequences
            mask = torch.ones(T, T, dtype=torch.bool).tril().expand(-1, -1)
            mask.masked_fill_(mask_indices > repeated_idx, False)
            # for m in range(len)
            # print(mask)
            self.packs[idx] = {
                "input_ids": i["input_ids"],
                "input_pos": i["input_pos"],
                "attention_mask": mask
            }

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "input_ids": [],
            "input_pos": [],
            "seq_lens": [],
        }

        # Only show progress bar on rank 0
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

        for sample in self.ds:
            input_ids = sample["input_ids"]
            # input_ids = sample

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(input_ids)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                    "Please set `split_across_pack=True` or increase `max_seq_len`."
                )

            # Update the current pack
            current_pack["input_ids"] += input_ids
            current_pack["input_pos"] += [x % self.max_seq_len for x in range(seq_len)]
            current_pack["seq_lens"] += [seq_len]

            # If the current pack is over the max_seq_len, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            while (
                len(current_pack["input_ids"]) > self.max_seq_len
                and not self._should_stop_packing()
            ):
                current_pack = self._split_and_add_pack(current_pack)

            if rank == 0:
                pbar.update()

            # Keep track of previous sample boundary
            self.previous_sample_boundary = len(current_pack["input_ids"])

            if self._should_stop_packing():
                break

        # Handle the last pack if there's leftover and we haven't filled up the max packs
        if len(current_pack["input_ids"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            # No need to handle splitting at this point so we can just add the current pack
            self._add_pack(current_pack)

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.max_seq_len
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "input_ids": current_pack["input_ids"][:boundary],
            "input_pos": current_pack["input_pos"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["input_ids"][boundary:])
            if self.split_across_pack
            else current_pack["seq_lens"][-1]
        )

        return {
            "input_ids": current_pack["input_ids"][boundary:],
            "input_pos": current_pack["input_pos"][boundary:],
            "seq_lens": [next_seq_len],
        }

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        # print(pack)
        # print(torch.tensor(pack["input_ids"], dtype=torch.long))
        # assert 0
        # print(type(pack["input_ids"]))
        return {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad input_ids
        num_padding_input_ids = self.max_seq_len - len(pack["input_ids"])
        padded_input_ids = F.pad(
            pack["input_ids"],
            (0, num_padding_input_ids),
            value=padding_idx,
        )

        # Add padding input_ids as a last seq len to ensure sum is max_seq_len
        padded_seq_lens = (
            torch.cat([pack["seq_lens"], torch.tensor([num_padding_input_ids])])
            if num_padding_input_ids > 0
            else pack["seq_lens"]
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "input_ids": padded_input_ids,
            "input_pos": padded_input_pos,
            # "seq_lens": padded_seq_lens,
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __iter__(self):
        for i in self.packs:
            yield i

    def return_dataloader(self, batch_size):
        return DataLoader(self.packs, batch_size=batch_size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.packs[idx]


gurks = PackedCucumbers("GurkLM-dev/data/val/", tokenizer)
dl = gurks.create_giant_dataloader()
torch.save(dl, 'tiny_dataloader_val.pth')

for i in dl:
    print(i)
    print(i["input_pos"].shape)
    break

