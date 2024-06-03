import numpy as np
from torch.utils.data import Dataset


class DatasetComposition(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self._lengths = [ds.__len__() for ds in datasets]
        self._cumsum_length = np.cumsum(self._lengths)
        self._len = sum(self._lengths)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        dataset_index = np.argwhere(self._cumsum_length > idx)[0][0]
        local_idx = idx % self._cumsum_length[dataset_index - 1]
        return self.datasets[dataset_index][local_idx]
