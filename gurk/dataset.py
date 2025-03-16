import os
import json
import random
import re
from typing import Dict, List, Optional

from datasets import Dataset
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm


class GurkFolderDataset:
    """
    Read in data from a custom gurk folder dataset.

    This dataset consists of data in a specific structure: The files are in
    .jsonlines format, containing pre-tokenized input of the right sequence
    length.

    Args:
        input_dir: path to the input files
    """

    def __init__(self, input_dir):
        self.data = self._read_data(input_dir)

    def _read_data(self, input_dir):
        """Read in individual files in order."""
        file_names = os.listdir(input_dir)
        file_names.sort(key=lambda x: int(re.match(r"^file(\d+)_.+$", x).group(1)))
        lines = []
        # Read in files one-by-one
        for file in file_names:
            with open(os.path.join(input_dir, file), encoding="utf-8") as file_in:
                for line in file_in:
                    lines.append(json.loads(line))
        return lines

    def get_dataloader(self, batch_size=32):
        """Create a dataloader object for the data from the directory.

        Args:
            batch_size (int): Batch size to use for the dataset creation.
                Defaults to 32.
        """
        dataset = Dataset.from_list(self.data)
        dataset.set_format(type='torch')
        return DataLoader(dataset, batch_size=batch_size)


class PackedCucumbers:
    """A class for packed dataset creation.

    Reads in raw text data from a directory, tokenizes it and returns a dataset
    packed to the defined max sequence length.

    Args:
        datapath: path to the directory containing the raw text files with the
            data.
        tokenizer (transformer.AutoTokenizer): the tokenizer to use
        seed (int): the seed to use for all random actions
        max_seq_len (int): the maximum sequence length to which to pack the
            sequences
        shuffle (bool): whether to shuffle the data or not. If set to True,
            all lines will be shuffled.            
    """

    def __init__(self, data_path, tokenizer, seed=13, max_seq_len=512, shuffle=True):
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.root = data_path
        # Get files and list of IDs
        self.files = {file_id: file_path for file_id, file_path in enumerate(os.listdir(data_path))}
        self.shuffeled_ids = list(self.files.keys())
        self.tokenizer = tokenizer

    def _load_all(self, file_paths):
        """Load all files to create a giant dataset.

        Args:
            file_paths (list[str]): list of filepaths

        Returns:
            PackedDataset: a dataset of the PackedDataset class
        """
        lines = []
        # Get the lines of text from the files
        for file_path in file_paths:
            with open(file_path, encoding="utf-8") as file_in:
                lines += [{"text": line} for line in file_in]
        # Shuffle lines if needed
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(lines)
        # Create dataset
        ds = Dataset.from_list(lines)
        ds = ds.map(self._tokenize)
        return PackedDataset(ds, max_seq_len=self.max_seq_len, split_across_pack=True)

    def _load(self, file_path):
        """Load a single file as a packed dataset.

        Args:
            file_paths (str): a filepath

        Returns:
            PackedDataset: a dataset of the PackedDataset class
        """
        with open(file_path, encoding="utf-8") as file_in:
            # Get the lines of text from the files
            lines = [{"text": line} for line in file_in]
            # Shuffle lines if needed
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(lines)
            # Create dataset
            ds = Dataset.from_list(lines)
            ds = ds.map(self._tokenize)
        return PackedDataset(ds, max_seq_len=self.max_seq_len, split_across_pack=True)

    def _tokenize(self, instance):
        """Tokenize an instance."""
        input_ids = self.tokenizer(instance["text"],
                                   return_attention_mask=False,
                                   add_special_tokens=False)["input_ids"]
        return {"input_ids": input_ids}

    def create_giant_dataset(self):
        """Create a dataset over all files from the provided input directory."""
        file_paths = [os.path.join(self.root, self.files[i]) for i in self.shuffeled_ids]
        return self._load_all(file_paths).packs

    def create_giant_dataloader(self, batch_size=32):
        """Create a dataloader over all files from the provided input directory."""
        file_paths = [os.path.join(self.root, self.files[i]) for i in self.shuffeled_ids]
        dataloader = DataLoader([i for f in file_paths for i in self._load(f).packs],
                                batch_size=batch_size)
        return dataloader

    def __len__(self):
        return len(self.ids)


class PackedDataset(Dataset):
    """Create a Dataset with packed sequences from a dataset of tokenized data.

    This Dataset class was taken from https://pytorch.org/torchtune/stable/_modules/torchtune/datasets/_packed.html#PackedDataset
    and has been modified to fit our needs in this project. Specifically,
    the original also packed labels throughout the dataset, which we don't have
    in our use-case. The alterations made are therefore in service of removing
    label handling from the code and only return the input IDs needed. For the
    individual altered functions, the docstrings have been modified to inform
    about the changes we made.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
            packs as possible.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
    """

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

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset.

        Note: This function is slightly altered, the original version contains
            labels in the pack as well, which have been removed in this version.
        """
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
        returns the start of the next pack.

        Note: This function is slightly altered, the original version contains
            labels in the pack as well, which have been removed in this version.
        """

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
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.

        Note: This function is slightly altered, the original version contains
            labels in the pack as well, which have been removed in this version.
        """
        return {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long).detach(),
            "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long).detach(),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long).detach(),
        }

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``.

        Note: This function is slightly altered, the original version contains
            labels in the pack as well, which have been removed in this version.
        """
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

        # Note: We altered this to return only the input ids we need for our task
        return {
            "input_ids": padded_input_ids,
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

