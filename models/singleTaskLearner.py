import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from config import config
from torch import nn, optim
from models.core_model import Net
from models.genericLearner import GenericLearner
from utils.custom_torch import zeros


class SingleTaskLearner(GenericLearner):
    def __init__(self, model_core: Net, input_length: int, output_length: int):
        super(SingleTaskLearner, self).__init__(
            model_core=model_core
        )  # Not sure what this does
        self.save_hyperparameters()
        # Hyperparameters
        self.learning_rate = config["stl_learning_rate"]
        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        self.log_vars = nn.Parameter(zeros((1)))
        # Heads
        self.prediction = nn.Linear(input_length, output_length)
        # Loss function
        self.loss_function = F.binary_cross_entropy

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
        self.metrics_update("train", prediction, prediction_label)
        return loss_prediction

    def validation_step(self, batch, _):
        prediction, prediction_label = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        self.log("loss_validate", loss_prediction)
        return loss_prediction

    def test_step(self, batch, _):
        prediction, prediction_label = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        self.metrics_update("test", prediction, prediction_label)
        self.log("Loss/test", loss_prediction)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def calculate_loss(self, prediction, correct_label):
        loss = self.loss_function(prediction, correct_label)
        return loss
