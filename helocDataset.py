import pandas as pd
import torch

from pandas import DataFrame, Series
from torch.utils.data import Dataset
from typing import Tuple

class HELOCDataset(Dataset):
    def __init__(self, dataset: DataFrame):
        dataset, predictor = self.split_predictor(dataset)
        self.values = torch.tensor(dataset.values, dtype=torch.float)
        self.predictors = predictor

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.predictors[index].unsqueeze(0)
        # return self.values[index]

    def split_predictor(self, data: DataFrame) -> Tuple[DataFrame, Series]:
        predictions_label = data.pop('RiskPerformance')
        normalized_prediction_labels = [0 if label == "Bad" else 1 for label in predictions_label]
        tensored_predictors = self.tensor_predictor(normalized_prediction_labels)
        return data, tensored_predictors

    def tensor_predictor(self, data: Series) -> torch.Tensor:
        #print(data)
        data = torch.tensor(data, dtype=torch.float)
        return data
