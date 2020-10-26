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

        # Metrics
        self.accuracy = pl.metrics.Accuracy()
        self.own_precision = pl.metrics.Precision() # Named because Trainer would try to overwrite own precision metric.
        self.recall = pl.metrics.Recall()
        self.fbeta = pl.metrics.Fbeta()

        # Hyperparameters
        self.learning_rate = 0.01

        # Pre-split model
        self.rest_of_model = model_core

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
        self.metrics_update("train-step", prediction, correct_label)
        return loss

    def training_epoch_end(self, outs):
        # Log epoch metric
        self.metrics_compute("train-epoch")

    def validation_step(self, batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.log('Loss/validate', loss)

    def test_step(self,batch, _):
        values, correct_label = batch
        prediction = self(values)
        loss = F.mse_loss(prediction, correct_label)
        self.metrics_update("test-step", prediction, correct_label)
        self.log('Loss/test', loss)

    def test_epoch_end(self, outs):
        self.metrics_compute("test-epoch")

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=self.learning_rate)

    def metrics_compute(self, label):
        self.log(f"Accuracy/{label}", self.accuracy.compute())
        self.log(f"Precision/{label}", self.own_precision.compute())
        self.log(f"Recall/{label}", self.recall.compute())
        self.log(f"Fbeta/{label}", self.fbeta.compute())

    def metrics_update(self, label, prediction, correct_label):
        self.log(f"Accuracy/{label}", self.accuracy(prediction, correct_label))
        self.log(f"Precision/{label}", self.own_precision(prediction, correct_label))
        self.log(f"Recall/{label}", self.recall(prediction, correct_label))
        self.log(f"Fbeta/{label}", self.fbeta(prediction, correct_label))

