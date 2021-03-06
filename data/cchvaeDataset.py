from data.helocDataset import HELOCDataset
from data.normalized_labels import Labels
import torch
import numpy as np
class CchvaeDataset(HELOCDataset):
    def __init__(self, dataset):
        dataset, explanations = self.split_explanation_label(dataset)
        self.prediction_data = Labels(dataset.iloc[:,1:].to_numpy())
        abs_explanation = torch.abs(torch.tensor(explanations)).numpy()
        self.explanation_labels = Labels(abs_explanation)
        self.explanation_labels.original_labels = torch.from_numpy(self.explanation_labels.original_labels)
        self.explanations = explanations
        super().__init__(dataset)

    def __getitem__(self, index):
        # return (
        #     self.prediction_data.original_labels[index],
        #     self.targets[index].unsqueeze(0),
        #     self.explanation_labels.original_labels[index]
        # )
        return (
            self.prediction_data.labels[index],
            self.targets[index].unsqueeze(0),
            self.explanation_labels.labels[index]
        )

    def split_explanation_label(self, data):
        counterfactual_label = data.pop("counterfactual delta")
        labels_as_list_entries = [
            list(map(float, label[1 : len(label) - 1].split(" ")))
            for label in counterfactual_label
        ]
        return data, labels_as_list_entries
