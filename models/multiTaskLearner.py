import torch.nn.functional as F
import torch

from config import config
from torch import nn, optim, autograd
from typing import Tuple
from models.core_model import Net
from models.genericLearner import GenericLearner
from utils.custom_torch import zeros, ones


def categorical_cross_entropy(explanation, true_explanation):
    return F.cross_entropy(explanation, torch.max(true_explanation, 1)[1])


class MultiTaskLearner(GenericLearner):
    def __init__(
        self, model_core: Net, input_length: int, output_length: Tuple[int, int]
    ):
        super(MultiTaskLearner, self).__init__(heads=2, model_core=model_core)
        self.save_hyperparameters()
        # Hyperparameters
        self.learning_rate = config["mtl_learning_rate"]
        # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
        # self.log_vars = nn.Parameter(torch.zeros((2)))
        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Linear(input_length, output_length[1])
        # Loss functions per head
        self.loss_functions = [F.mse_loss, categorical_cross_entropy]
        self.prediction_head.register_forward_hook(self.forward_hook)

    def forward(self, data_input):
        rest_output = self.rest_of_model(data_input)
        prediction = self.prediction_head(rest_output)
        prediction = torch.sigmoid(prediction)
        explanation = self.explanation_head(rest_output)
        explanation = F.softmax(explanation)
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
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        loss_explanation = self.calculate_loss(explanation, explanation_label, head=1)
        self.log("Loss/train-prediction", loss_prediction)
        self.log("Loss/train-explanation", loss_explanation)
        self.log("Loss/train-head-difference", loss_convergence)
        self.metrics_update("train-step", prediction, prediction_label)
        self.metrics_update("train-step", explanation, explanation_label, head=1)
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
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        loss_explanation = self.calculate_loss(explanation, explanation_label, head=1)
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
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        loss_explanation = self.calculate_loss(explanation, explanation_label, head=1)
        self.metrics_update("test-step", prediction, prediction_label)
        self.metrics_update("test-step", explanation, explanation_label, head=1)
        self.log("Loss/test", loss_prediction)

    def configure_optimizers(self):
        return optim.Adadelta(self.parameters(), lr=self.learning_rate)

    def calculate_loss(self, prediction, correct_label, head=0, T=0):
        loss_function = self.loss_functions[head]
        loss = loss_function(prediction, correct_label)
        precision = 1  # torch.exp(-self.log_vars[head])
        return precision * loss  # + self.log_vars[head] + T

    def converge_gradients(self, prediction, explanation):
        (layer_pred,) = autograd.grad(
            torch.sum(prediction, dim=1),
            self.act,
            grad_outputs=ones(prediction.shape[0]),
            create_graph=True,
        )
        (layer_exp,) = autograd.grad(
            torch.sum(explanation, dim=1),
            self.act,
            grad_outputs=ones(explanation.shape[0]),
            create_graph=True,
        )
        return self.get_explanation_prefix_difference(layer_pred, layer_exp)

    def converge_weights(self, explanation):
        highest_indices = torch.argmax(explanation, dim=1)
        counting_weights = torch.stack(
            [self.explanation_head.weight[index] for index in highest_indices]
        )
        return self.get_explanation_prefix_difference(
            self.prediction_head.weight, counting_weights
        )

    def get_explanation_prefix_difference(self, tensor1, tensor2):
        tensor1_positive = tensor1 > 0
        tensor1_negative = tensor1 < 0
        tensor2_positive = tensor2 > 0
        tensor2_negative = tensor2 < 0
        positive_xor = torch.logical_xor(tensor1_positive, tensor2_positive)
        negative_xor = torch.logical_xor(tensor1_negative, tensor2_negative)

        converge_distances = tensor2 * positive_xor.float()
        explanation_prefix_convergence_distance = F.mse_loss(
            converge_distances, zeros(converge_distances.size())
        )
        """
        explanation_prefix_convergence_distance = sum(
            sum(torch.abs(converge_distances))
        )
        """
        return explanation_prefix_convergence_distance
