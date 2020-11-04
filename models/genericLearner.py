from abc import abstractmethod
import pytorch_lightning as pl

from models.core_model import Net

class GenericLearner(pl.LightningModule):
    def __init__(self, model_core: Net, heads: int = 1):
        super(GenericLearner, self).__init__()
        self.rest_of_model = model_core
        self.save_hyperparameters()
        metrics = [
                pl.metrics.Accuracy(),
                pl.metrics.Precision(),
                pl.metrics.Recall(),
                pl.metrics.Fbeta(),
            ]
        self.metrics = [[*metrics] for head in range(heads)]

    @abstractmethod
    def forward(self, data_input): raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx): raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx): raise NotImplementedError

    @abstractmethod
    def test_step(self, batch, batch_idx): raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self): raise NotImplementedError

    @abstractmethod
    def calculate_loss(self): raise NotImplementedError

    def test_epoch_end(self, outs):
        self.metrics_compute("test-epoch")
    
    def training_epoch_end(self, outs):
        self.metrics_compute("train-epoch")

    def metrics_compute(self, label):
        for index, metric in enumerate(self.metrics):
            head = index + 1
            self.log(f"Accuracy/head-{head}/{label}", metric[0].compute())
            self.log(f"Precision/head-{head}/{label}", metric[0].compute())
            self.log(f"Recall/head-{head}/{label}", metric[0].compute())
            self.log(f"Fbeta/head-{head}/{label}", metric[0].compute())

    def metrics_update(self, label, prediction, correct_label):
        for index, metric in enumerate(self.metrics):
            head = index + 1
            self.log(f"Accuracy/head-{head}/{label}", metric[0](prediction, correct_label))
            self.log(f"Precision/head-{head}/{label}", metric[0](prediction, correct_label))
            self.log(f"Recall/head-{head}/{label}", metric[0](prediction, correct_label))
            self.log(f"Fbeta/head-{head}/{label}", metric[0](prediction, correct_label))
