import pandas as pd
import torch

from pandas import DataFrame
from torch.utils.data import Dataset

class HELOCDataset(Dataset):
    def __init__(self, dataset: DataFrame):
        self.values = torch.Tensor(dataset.values)
        indexed_labels = list(range(len(dataset.columns)))
        self.labels = torch.Tensor(indexed_labels)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels
        # return self.values[index]
