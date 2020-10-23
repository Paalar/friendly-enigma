import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from torch import nn, optim
from typing import Tuple
from model import Net

from print_colors import print_git_diff

class MultiTaskOutputWrapper(pl.LightningModule):
    def __init__(self, model_core: Net, input_length: int, output_length: Tuple[int, int]):
        super(MultiTaskOutputWrapper, self).__init__() # Not sure what this does
        self.rest_of_model = model_core

        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])

    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)

        # Heads
        #prediction = F.softmax(rest_output, dim=1)
        prediction = self.prediction_head(rest_output)
        # print("After last layer:", prediction)
        prediction = torch.sigmoid(prediction)
        # print("After sigmoid:", prediction)
        #print(f"Prediction head - {prediction}")
        # explanation = self.explanation_head(rest_output)
        # print(f"Prediction softmax - {prediction}")
        # explanation = F.softmax(explanation, dim=-1)
        return prediction

    def training_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        #criterion = nn.BCEWithLogitsLoss()
        print_git_diff(prediction, correct_label)
        loss = F.binary_cross_entropy(prediction, correct_label)
        #loss = criterion(prediction, correct_label)
        # print(f"Criterion - {loss}")
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        # print(f"Expected {prediction} to be {correct_label}")
        loss = F.binary_cross_entropy(prediction, correct_label)
        self.log('validation_loss', loss)

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=1)
