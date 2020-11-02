import torch
from helocDataset import HELOCDataset

class ExplanationDataset(HELOCDataset): 
    def __init__(self, dataset):
        dataset, predictor = self.split_predictor(dataset)
        self.predictors = predictor
        dataset, explanations = self.split_explanation_label(dataset)
        self.values = torch.tensor(dataset.values, dtype=torch.float)
        self.explanations = explanations

    def __getitem__(self, index):
        return self.values[index], self.predictors[index].unsqueeze(0), self.explanations[index]
        
    def split_explanation_label(self, data):
        explanation_label = data.pop('explanation set')
        return data, explanation_label


