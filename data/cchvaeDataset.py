import torch
from data.helocDataset import HELOCDataset
from utils.custom_torch import zeros


class CchvaeDataset(HELOCDataset):
    def __init__(self, dataset):
        dataset, explanations = self.split_explanation_label(dataset)
        self.explanations = explanations
        super().__init__(dataset)

    def __getitem__(self, index):
        return (
            self.values[index],
            self.predictors[index].unsqueeze(0),
            self.explanations[index],
        )

    def split_explanation_label(self, data):
        counterfactual_label = data.pop("counterfactual delta")
        labels_as_list_entries = [
            torch.tensor(list(map(float, label[1 : len(label) - 1].split(" "))))
            for label in counterfactual_label
        ]
        return data, labels_as_list_entries
