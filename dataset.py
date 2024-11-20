import os
import random

from torch.utils.data import IterableDataset

class GurkDataset(IterableDataset):

    def __init__(self, 
                 data_dir: str
                 ):
        super().__init__()
        self.root = data_dir
        self.ids = self._get_ids(data_dir=data_dir)

    def _get_ids(self, data_dir):
        return os.listdir(data_dir)

    def _read_file(self, path):
        content = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip().split()
                content.append(line)
        return content

    def __iter__(self):
        # random order of files
        random.shuffle(self.ids)
        for file_id in self.ids:
            path = os.path.join(self.root, file_id)
            file_content = self._read_file(path)
            random.shuffle(file_content)
            for data_point in file_content:
                yield data_point

    def __len__(self):
        return len(self.ids)

