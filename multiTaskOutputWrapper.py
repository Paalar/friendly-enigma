import torch.nn.functional as F

from torch import nn, optim
from typing import Tuple
from model import Net

class MultiTaskOutputWrapper(nn.Module):
    def __init__(self, model_core: Net, input_length: int, output_length: Tuple[int, int]):
        super(MultiTaskOutputWrapper, self).__init__() # Not sure what this does

        self.rest_of_model = model_core

        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0]) 
        self.explanation_head = nn.Linear(input_length, output_length[1])

    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)

        # Heads
        # print("output", rest_output)
        prediction = self.prediction_head(rest_output)
        # explanation = self.explanation_head(rest_output)
        # print("prediction head", prediction)
        prediction = F.softmax(prediction, dim=0)
        print("prediction softmax", prediction)
        # explanation = F.softmax(explanation, dim=-1)
        return prediction

    def train(self, dataset, predictors):
        prediction_criterion = nn.MSELoss()
        explanation_criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        for index, data in enumerate(dataset):
            print(predictors)
            values, labels = data
            prediction = self(values)
            prediction_loss = prediction_criterion(prediction, predictors[index])
            # explanation_loss = explanation_criterion(explanation, labels)
            optimizer.zero_grad() # Zeroes all gradient buffers of all parameters
            prediction_loss.backward()
            # explanation_loss.backward(retain_graph=True)
            optimizer.step()
