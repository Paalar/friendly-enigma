from abc import abstractmethod, ABC
import pytorch_lightning as pl
import torch
from utils.custom_torch import get_device
from config import config

from models.core_model import Net


class GenericLearner(pl.LightningModule, ABC):
    def __init__(self, model_core: Net, heads = [1]):
        super(GenericLearner, self).__init__()
        self.rest_of_model = model_core
        metrics = [
            pl.metrics.Accuracy,
            pl.metrics.Precision,
            pl.metrics.Recall,
        ]
        self.metrics = [[metric().to(get_device()) for metric in metrics] for head in range(len(heads))]
        self.heads = heads
        for index, head in enumerate(heads):
            self.metrics[index].append(pl.metrics.FBeta(num_classes=head).to(get_device()))
            self.metrics[index].append(pl.metrics.ConfusionMatrix(num_classes=head).to(get_device()))

    @abstractmethod
    def forward(self, data_input):
        raise NotImplementedError

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
        self.log(f"Confusion-matrix/head-{head}/{label}", metric[4].compute())

    def metrics_update(self, label, prediction, correct_label, head=0):
        metric = self.metrics[head]
        if head == 1:
            self.log(
                f"Accuracy/head-{head}/{label}",
                metric[0](prediction, torch.max(correct_label, 1)[1]),
            )
        else:
            self.log(
                f"Accuracy/head-{head}/{label}", metric[0](prediction, correct_label)
            )
        self.log(f"Precision/head-{head}/{label}", metric[1](prediction, correct_label))
        self.log(f"Recall/head-{head}/{label}", metric[2](prediction, correct_label))
        self.log(f"Fbeta/head-{head}/{label}", metric[3](prediction, correct_label))
        self.log(f"Confusion-matrix/head-{head}/{label}", metric[4](prediction, correct_label))
