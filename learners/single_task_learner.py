# Package imports
import pandas as pd
import pytorch_lightning as pl

# Subpackage
from pandas import DataFrame
from typing import Tuple
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


# def main():
#     logger = dashboard.create_logger()
#     data_module = HelocDataModule()
#     data_module.prepare_data()

#     # Instantiate model
#     nodes_before_split = 64
#     input_length = data_module.row_length

#     net = Net(input_length=input_length, output_length=nodes_before_split)

#     model = SingleTaskOutputWrapper(
#         model_core=net,
#         input_length=nodes_before_split,
#         output_length=1,
#     )

#     trainer = pl.Trainer(
#         max_epochs=13, logger=logger, checkpoint_callback=create_checkpoint_callback()
#     )
#     trainer.fit(model, data_module)
#     trainer.test(model, datamodule=data_module)


# if __name__ == "__main__":
#     main()
