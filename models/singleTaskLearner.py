import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from torch import nn, optim
from models.core_model import Net
from models.genericLearner import GenericLearner


class SingleTaskLearner(GenericLearner):
    def __init__(self, model_core: Net, input_length: int, output_length: int):
        super(SingleTaskLearner, self).__init__(
            model_core=model_core
        )  # Not sure what this does
        self.save_hyperparameters()
        # Hyperparameters
        self.learning_rate = 0.01
        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        self.log_vars = nn.Parameter(torch.zeros((1)))
        # Heads
        self.prediction = nn.Linear(input_length, output_length)
        # Loss function
        self.loss_function = F.mse_loss

    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)
        prediction = self.prediction(rest_output)
        prediction = torch.sigmoid(prediction)
        return prediction

    def predict_batch(self, batch):
        values, prediction_label = batch
        prediction = self(values)
        return prediction, prediction_label

    def training_step(self, batch, _):
        prediction, prediction_label = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        self.log("Loss/train-prediction", loss_prediction)
        self.metrics_update("train-step", prediction, prediction_label)
        return loss_prediction

    def validation_step(self, batch, _):
        prediction, prediction_label = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        self.log("loss_validate", loss_prediction)
        return loss_prediction

    def test_step(self, batch, _):
        prediction, prediction_label = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        self.metrics_update("test-step", prediction, prediction_label)
        self.log("Loss/test", loss_prediction)

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=self.learning_rate)

    def calculate_loss(self, prediction, correct_label):
        loss = self.loss_function(prediction, correct_label)
        precision = torch.exp(-self.log_vars)
        return precision * loss + self.log_vars
