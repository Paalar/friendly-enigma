import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from torch import nn, optim
from typing import Tuple
from model import Net


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

        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        self.log_vars = nn.Parameter(torch.zeros((2)))

        # Pre-split model
        self.rest_of_model = model_core

        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])

        # Loss functions per head
        self.loss_functions = [F.mse_loss, F.binary_cross_entropy]


    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)

        # Heads
        prediction = self.prediction_head(rest_output)
        prediction = torch.sigmoid(prediction)
        explanation = self.explanation_head(rest_output)
        explanation = torch.sigmoid(explanation)
        return prediction, explanation

    def training_step(self, batch, _):
        values, correct_label = batch
        prediction, explanation = self(values)
        loss_prediction = self.calculate_loss(prediction, correct_label, 0)
        loss_explanation = self.calculate_loss(explanation, correct_label, 1)
        self.log('Loss/train-prediction', loss_prediction)
        self.log('Loss/train-explanation', loss_explanation)
        self.metrics_update("train-step", prediction, correct_label)
        return loss_prediction+loss_explanation

    def training_epoch_end(self, outs):
        # Log epoch metric
        self.metrics_compute("train-epoch")

    def validation_step(self, batch, _):
        values, correct_label = batch
        prediction, explanation = self(values)
        loss_prediction = self.calculate_loss(prediction, correct_label, 0)
        loss_explanation = self.calculate_loss(explanation, correct_label, 1)
        self.log('Loss/validate', loss_prediction)
        return loss_prediction+loss_explanation

    def test_step(self,batch, _):
        values, correct_label = batch
        prediction, explanation = self(values)
        loss_prediction = self.calculate_loss(prediction, correct_label, 0)
        loss_explanation = self.calculate_loss(explanation, correct_label, 1)
        self.metrics_update("test-step", prediction, correct_label)
        self.log('Loss/test', loss_prediction)

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

    def calculate_loss(self, prediction, correct_label, head_number):
        loss_function = self.loss_functions[head_number]
        loss = loss_function(prediction, correct_label)
        precision = torch.exp(-self.log_vars[head_number])
        return precision * loss + self.log_vars[head_number]
