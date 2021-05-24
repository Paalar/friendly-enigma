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
        seed=None,
    ):
        self.max_epochs = max_epochs
        self.nodes_before_split = nodes_before_split
        self.data_module = data_module
        self.data_module.prepare_data()
        self.logger = dashboard.create_logger(
            cli_args=args, data_module=data_module, runner_type=checkpoints_prefix
        )
        self.checkpoints_prefix = checkpoints_prefix
        self.args = args
        self.seed = seed
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
        try:
            self.logger.experiment.log_table(
                "training.csv", tabular_data=self.data_module.training_split
            )
            self.logger.experiment.log_table(
                "test.csv", tabular_data=self.data_module.test_split
            )
            self.logger.experiment.log_table(
                "validation.csv", tabular_data=self.data_module.validate_split
            )
            self.logger.experiment.log_other("Seed", self.seed)
        except:
            # Some logging methods differ for each logger type. These are meant for Comet.ml. If they are not present, just surpress errors.
            pass


def create_checkpoint_callbacks(prefix):
    period_callback = ModelCheckpoint(
        period=(config["stl_epochs"] if prefix == "stl" else config["mtl_epochs"]) / 10,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}-recent",
        save_last=True,
    )
    loss_validate_callback = ModelCheckpoint(
        monitor="loss_validate",
        save_top_k=3,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}-top-validate",
    )
    return period_callback, loss_validate_callback
