import pandas as pd
import torch

from pandas import DataFrame, Series
from torch.utils.data import Dataset
from typing import Tuple

from data.normalized_labels import Labels

class HELOCDataset(Dataset):
    def __init__(self, predictors: DataFrame):
        predictors, target = self.split_predictor(predictors)
        self.predictors = torch.tensor(Labels(predictors.values).labels, dtype=torch.float)
        self.targets = target

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, index):
        return self.predictors[index], self.targets[index].unsqueeze(0)

    def split_predictor(self, data: DataFrame) -> Tuple[DataFrame, Series]:
        predictions_label = data.pop("RiskPerformance")
        normalized_prediction_labels = [
            0 if label == "Bad" else 1 for label in predictions_label
        ]
        tensored_targets = self.tensor_predictor(normalized_prediction_labels)
        return data, tensored_targets

    def tensor_predictor(self, data: Series) -> torch.Tensor:
        data = torch.tensor(data, dtype=torch.float)
        return data
