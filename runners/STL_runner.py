# Package imports
import pytorch_lightning as pl

# Subpackage
from functools import partialmethod
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Project imports
from config import config
from models.core_model import Net
from models.singleTaskLearner import SingleTaskLearner
from data.helocDataModule import HelocDataModule
from utils import dashboard


class STLRunner:
    def __init__(
        self,
        nodes_before_split=config["nodes_before_split"],
        max_epochs=config["stl_epochs"],
        data_module=HelocDataModule(),
        checkpoints_prefix="stl",
        tune_config=None,
        args=None,
    ):
        self.max_epochs = max_epochs
        self.nodes_before_split = nodes_before_split
        self.logger = dashboard.create_logger()
        self.data_module = data_module
        self.data_module.prepare_data()
        self.checkpoints_prefix = checkpoints_prefix
        self.args = args
        input_length = self.data_module.row_length
        self.model_core = Net(
            input_length=input_length,
            output_length=nodes_before_split,
            tune_config=tune_config,
        )
        self.model = SingleTaskLearner(
            model_core=self.model_core,
            input_length=nodes_before_split,
            output_length=1,
        )

    def run(self):
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            max_epochs=self.max_epochs,
            logger=self.logger,
            callbacks=[*create_checkpoint_callbacks(self.checkpoints_prefix)],
        )
        trainer.fit(self.model, self.data_module)
        trainer.test(self.model, datamodule=self.data_module)
        # dashboard.create_confusion_matrix(self.model, self.logger, self.data_module)


def create_checkpoint_callbacks(prefix):
    period_callback = ModelCheckpoint(
        period=(config['stl_epochs'] if prefix == "stl" else config["mtl_epochs"])/10,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}-period",
    )
    loss_validate_callback = ModelCheckpoint(
        monitor="loss_validate",
        save_top_k=3,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}-top-validate",
    )
    accuracy_epoch_callback = ModelCheckpoint(
        monitor="Accuracy/head-1/train-step",
        save_top_k=3,
        mode="max",
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename="heloc-{epoch:02d}-{Accuracy/head-1/train-step:.2f}-top-accuarcy-expl",
    )
    return period_callback, loss_validate_callback, accuracy_epoch_callback
