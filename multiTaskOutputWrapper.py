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
        self.accuracy = pl.metrics.Accuracy()
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
        loss = F.mse_loss(prediction, correct_label)
        #loss = criterion(prediction, correct_label)
        self.log('Loss/train', loss)
        self.log('Accuracy/train-step', self.accuracy(prediction, correct_label))
        return loss
    def training_epoch_end(self, outs):
        # Log epoch metric
        self.log("Accuracy/train-epoch", self.accuracy.compute())

    def validation_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        # print(f"Expected {prediction} to be {correct_label}")
        loss = F.mse_loss(prediction, correct_label)
        self.log('Loss/validate', loss)
    
    def test_step(self,batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.log('Accuracy/test-step', self.accuracy(prediction, correct_label))
        self.log('Accuracy/test-epoch', self.accuracy.compute())
        self.log('Loss/test', loss)


    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=0.01)
