from abc import abstractmethod, ABC

import pytorch_lightning as pl
import torch
import torchmetrics as tm

from utils.custom_torch import get_device
from models.core_model import Net


class GenericLearner(pl.LightningModule, ABC):
    def __init__(self, model_core: Net, num_classes: int = [1]):
        super(GenericLearner, self).__init__()
        self.rest_of_model = model_core
        metrics = [
            tm.Accuracy,
            tm.Precision,
            tm.Recall,
        ]
        self.metrics = [[metric().to(get_device()) for metric in metrics] for head in range(len(num_classes))]
        self.heads = len(num_classes)
        for index, head in enumerate(num_classes):
            self.metrics[index].append(pl.metrics.FBeta(num_classes=head).to(get_device()))
            # self.metrics[index].append(pl.metrics.ConfusionMatrix(num_classes=2 if head == 1 else head))

    @abstractmethod
    def forward(self, data_input):
        raise NotImplementedError

    def predict(self, data_input):
        data_input_torch = torch.tensor(data_input, dtype=torch.float)
        return self.forward(data_input_torch).int()

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(self):
        raise NotImplementedError

    def test_epoch_end(self, outs):
        for head in range(self.heads):
            self.metrics_compute("test-epoch", head=head)

    def training_epoch_end(self, outs):
        for head in range(self.heads):
            self.metrics_compute("train-epoch", head=head)

    def metrics_compute(self, label, head):
        metric = self.metrics[head]
        self.log(f"Accuracy/head-{head}/{label}", metric[0].compute())
        self.log(f"Precision/head-{head}/{label}", metric[1].compute())
        self.log(f"Recall/head-{head}/{label}", metric[2].compute())
        self.log(f"Fbeta/head-{head}/{label}", metric[3].compute())
        # metric[4].compute()

    def metrics_update(self, label, prediction, target, head=0):
        metric = self.metrics[head]
        if head == 1:
            prediction = torch.exp(prediction)
            self.log(
                f"Accuracy/head-{head}/{label}",
                metric[0](prediction, torch.max(target, 1)[1]),
            )
            self.log(f"Precision/head-{head}/{label}", metric[1](prediction, torch.max(target, 1)[1]))
            self.log(f"Recall/head-{head}/{label}", metric[2](prediction, torch.max(target, 1)[1]))
            self.log(f"Fbeta/head-{head}/{label}", metric[3](prediction, torch.max(target, 1)[1]))
        else:
            target = target.to(torch.int)
            self.log(
                f"Accuracy/head-{head}/{label}", metric[0](prediction, target)
            )
            self.log(f"Precision/head-{head}/{label}", metric[1](prediction, target))
            self.log(f"Recall/head-{head}/{label}", metric[2](prediction, target))
            self.log(f"Fbeta/head-{head}/{label}", metric[3](prediction, target))
        # metric[4].update(prediction, correct_label)
        # metric[4](prediction, correct_label)

    def print_gradients(self):
        for name, parameter in self.named_parameters():
            gradient = parameter.grad
            if gradient == None:
                print(f"Parameter {name} is none")
                continue
            pred = torch.where(gradient == 0, 0,1)
            if torch.sum(pred) == 0:
                print(f"Index {self.current_epoch} - Parameter {name}'s gradients are summed to 0.")

    # def on_before_zero_grad(self, optimizer) -> None:
    #     self.print_gradients()
