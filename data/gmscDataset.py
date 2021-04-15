import pandas as pd
import torch
import ast

from pandas import DataFrame, Series
from torch.utils.data import Dataset
from typing import Tuple

from data.normalized_labels import Labels

class GMSCDataset(Dataset):
    def __init__(self, dataset: DataFrame):
        dataset, explanations = self.split_explanation_label(dataset)
        self.prediction_data = Labels(dataset.iloc[:,1:].to_numpy())
        abs_explanation = torch.abs(torch.tensor(explanations)).numpy()
        self.explanation_labels = Labels(abs_explanation)
        self.explanation_labels.original_labels = torch.from_numpy(self.explanation_labels.original_labels)
        self.explanations = explanations

        predictors, targets = self.split_predictor(dataset)
        self.predictors = torch.tensor(Labels(predictors.values).labels, dtype=torch.float)
        self.targets = targets

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, index):
        # return self.predictors[index], self.targets[index].unsqueeze(0)
        return (
            self.prediction_data.labels[index],
            self.targets[index].unsqueeze(0),
            self.explanation_labels.labels[index]
        )

    def split_predictor(self, data: DataFrame) -> Tuple[DataFrame, Series]:
        predictions = self.tensor_predictor(data.pop("SeriousDlqin2yrs").to_list())
        return data, predictions

    def tensor_predictor(self, data: Series) -> torch.Tensor:
        data = torch.tensor(data, dtype=torch.float)
        return data
    
    def split_explanation_label(self, data):
        counterfactual_label = data.pop("counterfactual delta")
        labels_as_list_entries = []
        for index, value in counterfactual_label.items():
            value = value.strip("[]").split()
            value = list(map(float, value))
            labels_as_list_entries.append(value)
        return data, labels_as_list_entries
