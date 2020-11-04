import torch
from data.helocDataset import HELOCDataset


class ExplanationDataset(HELOCDataset):
    def __init__(self, dataset):
        dataset, explanations = self.split_explanation_label(dataset)
        self.explanations = explanations
        print("Explanation", explanations)
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
        explanation_data = [
            self.create_explanation_as_ints(
                column_labels.index(label[3 : len(label) - 3])
            )
            for label in explanation_label
        ]
        return data, explanation_data

    def create_explanation_as_ints(self, index):
        template = torch.zeros([1, 23])
        template[0][index] = 1
        return template
