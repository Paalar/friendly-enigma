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
        metrics = [tm.Accuracy, tm.Precision, tm.Recall]
        self.metrics = [
            [metric().to(get_device()) for metric in metrics]
            for head in range(len(num_classes))
        ]
        self.heads = len(num_classes)
        for index, head in enumerate(num_classes):
            self.metrics[index].append(
                pl.metrics.FBeta(num_classes=head).to(get_device())
            )
            # self.metrics[index].append(tm.AUROC(num_classes=head))
            # self.metrics[index].append(pl.metrics.ConfusionMatrix(num_classes=2 if head == 1 else head))
        self.metrics = {
            "test": self.instantiate_metrics("test", num_classes),
            "train": self.instantiate_metrics("train", num_classes),
        }
        self.train_head2_AUROC = tm.AUROC(num_classes=10)

    def instantiate_metrics(self, label, num_classes):
        heads = len(num_classes)
        metrics = [tm.Accuracy, tm.Precision, tm.Recall]
        metrics_per_head = [{} for head in range(heads)]
        for index, head in enumerate(metrics_per_head):
            for metric in metrics:
                try:
                    head.update(
                        {
                            f"{metric.__name__}/head-{index}/{label}": metric(
                                num_classes=num_classes[index]
                            )
                        }
                    )
                except TypeError:
                    head.update({f"{metric.__name__}/head-{index}/{label}": metric()})

                head[f"FBeta/head-{index}/{label}"] = tm.FBeta(
                    num_classes=num_classes[index]
                )
                if index == 0:
                    auroc = tm.AUROC()
                    auroc.reorder = True
                    head[f"AUROC/head-{index}/{label}"] = auroc
        metrics_as_MetricCollection = [
            tm.MetricCollection(head) for head in metrics_per_head
        ]
        for collection in metrics_as_MetricCollection:
            collection.persistent()
        return metrics_as_MetricCollection

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
            self.metrics_compute("test", head=head)

    def training_epoch_end(self, outs):
        for head in range(self.heads):
            self.metrics_compute("train", head=head)

    def metrics_compute(self, label, head):
        metric = self.metrics[label][head]
        try:
            self.log_dict(metric.compute())
        except ValueError as e:
            print(e)
            print("Continuing..")

    def metrics_update(self, label, prediction, target, head=0):
        metric = self.metrics[label][head]
        if head == 1:
            prediction = torch.exp(prediction).detach()
            target = torch.max(target.detach(), 1)[1]
            metric.update(prediction, target)
        else:
            target = target.to(torch.int).detach()
            prediction = prediction.detach()
            metric.update(prediction, target)

    def print_gradients(self):
        for name, parameter in self.named_parameters():
            gradient = parameter.grad
            if gradient == None:
                print(f"Parameter {name} is none")
                continue
            pred = torch.where(gradient == 0, 0, 1)
            if torch.sum(pred) == 0:
                print(
                    f"Index {self.current_epoch} - Parameter {name}'s gradients are summed to 0."
                )

    # def on_before_zero_grad(self, optimizer) -> None:
    #     self.print_gradients()
