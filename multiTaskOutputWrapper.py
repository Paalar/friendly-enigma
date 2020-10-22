import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from torch import nn, optim
from typing import Tuple
from model import Net

class MultiTaskOutputWrapper(pl.LightningModule):
    def __init__(self, model_core: Net, input_length: int, output_length: Tuple[int, int]):
        super(MultiTaskOutputWrapper, self).__init__() # Not sure what this does

        self.rest_of_model = model_core

        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0]) 
        self.explanation_head = nn.Linear(input_length, output_length[1])

    def forward(self, data_input):
        #print("Input", data_input)
        rest_output = self.rest_of_model(data_input)
        #print("Prelastlayer", rest_output)

        # Heads
        # print("output", rest_output)
        prediction = self.prediction_head(rest_output)
        # explanation = self.explanation_head(rest_output)
        #print("prediction head", prediction)
        prediction = F.softmax(prediction, dim=0)
        #print("prediction softmax", prediction)
        # explanation = F.softmax(explanation, dim=-1)
        return prediction

    def training_step(self, batch, batch_idx):
        values, correct_label = batch

    def train(self, dataset, predictors):
        prediction_criterion = nn.BCELoss()
        explanation_criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        for epoch in range(3):
            for index, data in enumerate(dataset):
                print(f"Training iteration {index}")
                optimizer.zero_grad() # Zeroes all gradient buffers of all parameters
                #print(predictors)
                values, labels = data
                #print("Values", values)
                #print("Labels", labels)
                prediction = self(values)
                #print("Predictors", predictors)
                #print("PredDim", prediction.size())
                #print("PredLabelDim", predictors[index].unsqueeze_(0))
                correct_prediction = predictors[index].unsqueeze(0)
                prediction_loss = prediction_criterion(prediction, correct_prediction)
                # explanation_loss = explanation_criterion(explanation, labels)
                prediction_loss.backward()
                print("Loss", prediction_loss)
                # explanation_loss.backward(retain_graph=True)
                optimizer.step()
