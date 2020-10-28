# Package imports
import pandas as pd
import pytorch_lightning as pl
import os

# Subpackage
from pandas import DataFrame
from typing import Tuple
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from datetime import datetime

# Project imports
from model import Net
from multiTaskOutputWrapper import MultiTaskOutputWrapper
from helocDataModule import HelocDataModule

def main():
    data_module = HelocDataModule()
    data_module.prepare_data()

    # Configure logging
    api_key = os.environ.get('COMET_API_KEY')
    logger = TensorBoardLogger('lightning_logs')
    today = datetime.today()
    if api_key:
        logger = CometLogger(
            api_key=api_key,
            project_name='master-jk-pl',
            experiment_name=today.strftime("%d/%m/%y - %H:%M")
        )
    else:
        print("No Comet-API-key found, defaulting to Tensorboard", flush=True)

    # Instantiate model
    nodes_before_split = 64
    input_length = data_module.row_length
    net = Net(input_length=input_length, output_length=nodes_before_split)
    model = MultiTaskOutputWrapper(model_core=net, input_length=nodes_before_split, output_length=(1,1))
    trainer = pl.Trainer(max_epochs=150, logger=logger)
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
