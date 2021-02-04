import torch
from data.helocDataset import HELOCDataset
from utils.custom_torch import zeros


class ExplanationDataset(HELOCDataset):
    def __init__(self, dataset):
        dataset, explanations = self.split_explanation_label(dataset)
        self.explanations = explanations
        super().__init__(dataset)
        # dataset, predictor = self.split_predictor(dataset)
        # self.predictors = predictor
        # self.values = torch.tensor(dataset.values, dtype=torch.float)

    def __getitem__(self, index):
        return (
            self.values[index],
            self.predictors[index].unsqueeze(0),
            self.explanations[index],
        )

    def split_explanation_label(self, data):
        explanation_label = data.pop("explanation set")
        column_labels = data.columns[1:].tolist()
        labels_as_list_entries = [
            label[2 : len(label) - 2].replace("'", "").replace(" ", "").split(",")
            for label in explanation_label
        ]
        labels_indices = [
            [column_labels.index(label) for label in label_list]
            for label_list in labels_as_list_entries
        ]
        explanation_data = [
            self.create_explanation_as_ints(indices) for indices in labels_indices
        ]
        return data, explanation_data

    def create_explanation_as_ints(self, indices):
        template = zeros([1, 23])
        for index in indices:
            template[0][index] = 1
        return template[0]
