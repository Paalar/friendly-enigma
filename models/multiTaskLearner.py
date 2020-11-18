import torch.nn.functional as F
import torch

from config import config
from torch import nn, optim, autograd
from typing import Tuple
from models.core_model import Net
from models.genericLearner import GenericLearner


class MultiTaskLearner(GenericLearner):
    def __init__(
        self, model_core: Net, input_length: int, output_length: Tuple[int, int]
    ):
        super(MultiTaskLearner, self).__init__(heads=2, model_core=model_core)
        self.save_hyperparameters()
        # Hyperparameters
        self.learning_rate = config["mtl_learning_rate"]
        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        self.log_vars = nn.Parameter(torch.zeros((2)))
        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])
        # Loss functions per head
        self.loss_functions = [F.mse_loss, F.mse_loss]
        self.prediction_head.register_forward_hook(self.forward_hook)

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

    def forward_hook(self, module, in_data, output):
        self.act = in_data

    def training_step(self, batch, _):
        (
            prediction,
            prediction_label,
            explanation,
            explanation_label,
        ) = self.predict_batch(batch)

        loss_convergence = (
            self.converge_gradients(prediction, explanation)
            if config["loss_converge_method"] == "gradients"
            else self.converge_weights(explanation)
        )
        loss_prediction = self.calculate_loss(prediction, prediction_label, 0)
        loss_explanation = self.calculate_loss(explanation, explanation_label, 1)
        self.log("Loss/train-prediction", loss_prediction)
        self.log("Loss/train-explanation", loss_explanation)
        self.log("Loss/train-head-difference", loss_convergence)
        self.metrics_update("train-step", prediction, prediction_label)
        self.metrics_update("train-step", prediction, prediction_label, head=1)
        pred_weight = (
            1  # 0.2 if self.current_epoch > 100 else 50 / (self.current_epoch + 1)
        )
        alignment_weight = (
            1  # (1 if self.current_epoch > 100 else 200 / (self.current_epoch + 1))
        )
        return (
            pred_weight * (loss_prediction + loss_explanation)
            + alignment_weight * loss_convergence
        )

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

    def calculate_loss(self, prediction, correct_label, head_number, T=0):
        loss_function = self.loss_functions[head_number]
        loss = loss_function(prediction, correct_label)
        precision = 1  # torch.exp(-self.log_vars[head_number])
        return precision * loss  # + self.log_vars[head_number] + T

    def converge_gradients(self, prediction, explanation):
        (layer_pred,) = autograd.grad(
            torch.sum(prediction, dim=1),
            self.act,
            grad_outputs=torch.ones(prediction.shape[0]),
            create_graph=True,
        )
        (layer_exp,) = autograd.grad(
            torch.sum(explanation, dim=1),
            self.act,
            grad_outputs=torch.ones(explanation.shape[0]),
            create_graph=True,
        )
        positive_xor, negative_xor = self.get_prefix_differences(layer_pred, layer_exp)
        loss_convergence = sum(
            [
                abs(value) if positive_xor[index][value_index] else 0
                for (index, batch_item) in enumerate(layer_exp)
                for (value_index, value) in enumerate(batch_item)
            ]
        )
        return loss_convergence

    def converge_weights(self, explanation):
        highest_index = torch.argmax(explanation[0]).item()
        counting_weights = self.explanation_head.weight[highest_index]
        positive_xor, negative_xor = self.get_prefix_differences(
            self.prediction_head.weight, counting_weights
        )
        loss_convergence = sum(
            [
                abs(x) if positive_xor[0][index] else 0
                for (index, x) in enumerate(counting_weights)
            ]
        )
        return loss_convergence

    def get_prefix_differences(self, tensor1, tensor2):
        tensor1_positive = tensor1 > 0
        tensor1_negative = tensor1 < 0
        tensor2_positive = tensor2 > 0
        tensor2_negative = tensor2 < 0
        positive_xor = torch.logical_xor(tensor1_positive, tensor2_positive)
        negative_xor = torch.logical_xor(tensor1_negative, tensor2_negative)
        return positive_xor, negative_xor
