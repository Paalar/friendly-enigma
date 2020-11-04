# Package imports
import pytorch_lightning as pl

# Subpackage
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

# Project imports
from model import Net
from singleTaskOutputWrapper import SingleTaskOutputWrapper
from data.helocDataModule import HelocDataModule
import dashboard

class SingleTaskLearner():
    def __init__(self, nodes_before_split=64, max_epochs=13, data_module = HelocDataModule()):
        self.max_epochs=max_epochs
        self.nodes_before_split=nodes_before_split
        self.logger = dashboard.create_logger()
        self.data_module = data_module
        self.data_module.prepare_data()
        input_length=self.data_module.row_length
        self.model_core = Net(input_length=input_length, output_length=nodes_before_split)
        self.model = SingleTaskOutputWrapper(
            model_core=self.model_core,
            input_length=nodes_before_split,
            output_length=1,
        )
    
    def run(self):
        trainer = pl.Trainer(
            max_epochs=13, logger=self.logger, checkpoint_callback=create_checkpoint_callback()
        )
        trainer.fit(self.model, self.data_module)
        trainer.test(self.model, datamodule=self.data_module)


def create_checkpoint_callback():
    return ModelCheckpoint(
        monitor="loss_validate",
        save_top_k=3,
        dirpath=f"./checkpoints/{datetime.now().isoformat()}",
        filename="heloc-{epoch:02d}-{loss_validate:.2f}",
    )
