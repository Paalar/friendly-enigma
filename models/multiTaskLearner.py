import torch.nn.functional as F
import torch
from torch.utils import data

from config import config
from torch import nn, optim, autograd
from typing import Tuple
from models.core_model import Net
from models.genericLearner import GenericLearner
from utils.custom_torch import zeros, ones


def categorical_cross_entropy(explanation, true_explanation):
    return F.cross_entropy(explanation, torch.max(true_explanation, 1)[1])


def nll(explanation, true_explanation):
    w = torch.tensor([0.,0.,7.,0.,7.,2.,2.,6.,6.,2.,3.,7.,4.,7.,6.,4.,4.,7.,7.,5.,4.,3.,7.])
    return nn.NLLLoss(weight=w)(explanation, torch.max(true_explanation, 1)[1])


class MultiTaskLearner(GenericLearner):
    def __init__(
        self, model_core: Net, input_length: int, output_length: Tuple[int, int]
    ):
        super(MultiTaskLearner, self).__init__(
            num_classes=[1, 23], model_core=model_core
        )
        self.save_hyperparameters()
        # Hyperparameters
        self.learning_rate = config["mtl_learning_rate"]
        # Heads
        self.prediction_head = nn.Linear(input_length, output_length[0])
        self.explanation_head = nn.Sequential(
            nn.Linear(input_length, 250),
            nn.ReLU(),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Linear(125, output_length[1]),
            nn.ReLU()
        )
        # Loss functions per head
        self.loss_functions = [F.binary_cross_entropy, nll]
        self.prediction_head.register_forward_hook(self.forward_hook)

    def forward(self, data_input):
        data_input = data_input.to(torch.float32) # C-CHVAE
        rest_output = self.rest_of_model(data_input)
        prediction = self.prediction_head(rest_output)
        prediction = torch.sigmoid(prediction)
        explanation = self.explanation_head(rest_output)
        explanation = F.softmax(explanation, dim=-1)
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
        # self.log("Loss/train-head-difference", loss_convergence)
        self.metrics_update("train-step", prediction, prediction_label)
        self.metrics_update("train-step", explanation, explanation_label, head=1)
        pred_weight = (
            0.2  # 0.2 if self.current_epoch > 100 else 50 / (self.current_epoch + 1)
        )
        alignment_weight = (
            0.8  # (1 if self.current_epoch > 100 else 200 / (self.current_epoch + 1))
        )
        return self.total_loss(loss_prediction=loss_prediction, loss_explanation=loss_explanation)
        # return (
        #     pred_weight * (loss_prediction + loss_explanation)
        #     + alignment_weight * loss_convergence
        # )


    def validation_step(self, batch, _):
        (
            prediction,
            prediction_label,
            explanation,
            explanation_label,
        ) = self.predict_batch(batch)
        loss_prediction = self.calculate_loss(prediction, prediction_label)
        loss_explanation = self.calculate_loss(explanation, explanation_label, head=1)
        loss = self.total_loss(loss_prediction=loss_prediction, loss_explanation=loss_explanation)
        self.log("loss_validate", loss, prog_bar=True)

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

    def calculate_loss(self, prediction, correct_label, head=0):
        loss_function = self.loss_functions[head]
        loss = loss_function(prediction, correct_label)
        return loss

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
            [self.explanation_head.weight[index] for index in highest_indices] # Does not work with weights
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

        converge_distances = (tensor2 * positive_xor.float() + tensor2 * negative_xor.float()) / 2
        explanation_prefix_convergence_distance = F.mse_loss(
            converge_distances, zeros(converge_distances.size())
        )
        """
        explanation_prefix_convergence_distance = sum(
            sum(torch.abs(converge_distances))
        )
        """
        return explanation_prefix_convergence_distance

    def total_loss(self, loss_prediction, loss_explanation):
        return loss_explanation
        # return (
        #     loss_prediction + loss_explanation
        # ) / 2
