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
        self.own_precision = pl.metrics.Precision()
        self.rest_of_model = model_core
        self.learning_rate = 0.01

        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])


    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)

        # Heads
        prediction = self.prediction_head(rest_output)
        prediction = torch.sigmoid(prediction)
        #explanation = self.explanation_head(rest_output)
        #explanation = F.softmax(explanation, dim=-1)
        return prediction

    def training_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.log('Loss/train', loss)
        self.log('Accuracy/train-step', self.accuracy(prediction, correct_label))
        self.log('Precision/train-step', self.own_precision(prediction, correct_label))
        return loss

    def training_epoch_end(self, outs):
        # Log epoch metric
        self.log("Accuracy/train-epoch", self.accuracy.compute())
        self.log('Precision/train-epoch', self.own_precision.compute())

    def validation_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.log('Loss/validate', loss)

    def test_step(self,batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.log('Accuracy/test-step', self.accuracy(prediction, correct_label))
        self.log('Precision/test-step', self.own_precision(prediction, correct_label))
        self.log('Loss/test', loss)

    def test_epoch_end(self, outs):
        self.log('Accuracy/test-epoch', self.accuracy.compute())
        self.log('Precision/test-epoch', self.own_precision.compute())

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=self.learning_rate)
