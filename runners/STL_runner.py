# Package imports
import pytorch_lightning as pl

# Subpackage
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

# Project imports
from models.core_model import Net
from models.singleTaskLearner import SingleTaskLearner
from data.helocDataModule import HelocDataModule
import dashboard


class STLRunner:
    def __init__(
        self,
        nodes_before_split=64,
        max_epochs=13,
        data_module=HelocDataModule(),
        checkpoints_prefix="stl",
    ):
        self.max_epochs = max_epochs
        self.nodes_before_split = nodes_before_split
        self.logger = dashboard.create_logger()
        self.data_module = data_module
        self.data_module.prepare_data()
        self.checkpoints_prefix = checkpoints_prefix
        input_length = self.data_module.row_length
        self.model_core = Net(
            input_length=input_length, output_length=nodes_before_split
        )
        self.model = SingleTaskLearner(
            model_core=self.model_core,
            input_length=nodes_before_split,
            output_length=1,
        )

    def run(self):
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=self.logger,
            checkpoint_callback=create_checkpoint_callback(self.checkpoints_prefix),
        )
        trainer.fit(self.model, self.data_module)
        trainer.test(self.model, datamodule=self.data_module)


def create_checkpoint_callback(prefix):
    return ModelCheckpoint(
        monitor="loss_validate",
        save_top_k=3,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H:%M:%S')}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}",
    )
