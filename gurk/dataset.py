import os
import random

from torch.utils.data import IterableDataset

class GurkDataset(IterableDataset):

    def __init__(self, 
                 data_dir: str,
                 shuffle=True

                 ):
        super().__init__()
        self.root = data_dir
        self.ids = self._get_ids(data_dir=data_dir)
        self.seed = None
        self.shuffle = shuffle

    def _get_ids(self, data_dir):
        return os.listdir(data_dir)

    def _read_file(self, path):
        content = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip().split()
                content.append(line)
        return content
    
    def permute(self, seed=None):
        if self.shuffle is False:
            raise ValueError("Can't permute data set if self.shuffle is False.")
        if seed is None:
            seed = random.randint(0, 999999)
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.ids)

    def __iter__(self):
        for file_id in self.ids:
            path = os.path.join(self.root, file_id)
            file_content = self._read_file(path)
            if self.seed is not None and self.shuffle:
                random.seed(self.seed)
                random.shuffle(file_content)
            for data_point in file_content:
                yield data_point

    def __len__(self):
        return len(self.ids)

