import torch.nn.functional as F
import torch

from torch import nn, optim
from typing import Tuple
from models.core_model import Net
from models.genericLearner import GenericLearner


class MultiTaskLearner(GenericLearner):
    def __init__(
        self, model_core: Net, input_length: int, output_length: Tuple[int, int]
    ):
        super(MultiTaskLearner, self).__init__(heads=2, model_core=model_core)
        # Hyperparameters
        self.learning_rate = 0.01
        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        self.log_vars = nn.Parameter(torch.zeros((2)))
        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])
        # Loss functions per head
        self.loss_functions = [F.mse_loss, F.mse_loss]

    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)
        prediction = self.prediction_head(rest_output)
        prediction = torch.sigmoid(prediction)
        explanation = self.explanation_head(rest_output)
        explanation = torch.sigmoid(explanation)
        return prediction, explanation

    def predict_batch(self, batch):
        values, prediction_label, explanation_label = batch
        prediction, explanation = self(values)
        return prediction, prediction_label, explanation, explanation_label

    def training_step(self, batch, _):
        (
            prediction,
            prediction_label,
            explanation,
            explanation_label,
        ) = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label, 0)
        loss_explanation = self.calculate_loss(explanation, explanation_label, 1)
        self.log("Loss/train-prediction", loss_prediction)
        self.log("Loss/train-explanation", loss_explanation)
        self.metrics_update("train-step", prediction, prediction_label)
        self.metrics_update("train-step", prediction, prediction_label, head=1)
        return loss_prediction + loss_explanation

    def validation_step(self, batch, _):
        (
            prediction,
            prediction_label,
            explanation,
            explanation_label,
        ) = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label, 0)
        loss_explanation = self.calculate_loss(explanation, explanation_label, 1)
        self.log("loss_validate", loss_prediction)
        self.log("loss_validate", loss_prediction)
        return loss_prediction + loss_explanation

    def test_step(self, batch, _):
        (
            prediction,
            prediction_label,
            explanation,
            explanation_label,
        ) = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label, 0)
        loss_explanation = self.calculate_loss(explanation, explanation_label, 1)
        self.metrics_update("test-step", prediction, prediction_label)
        self.metrics_update("test-step", explanation, explanation_label, head=1)
        self.log("Loss/test", loss_prediction)

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=self.learning_rate)

    def calculate_loss(self, prediction, correct_label, head_number):
        loss_function = self.loss_functions[head_number]
        loss = loss_function(prediction, correct_label)
        precision = torch.exp(-self.log_vars[head_number])
        return precision * loss + self.log_vars[head_number]
