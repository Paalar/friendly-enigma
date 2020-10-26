# Package imports
import pandas as pd
import pytorch_lightning as pl
import os

# Subpackage
from torch.utils.data import DataLoader
from pandas import DataFrame
from typing import Tuple
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

# Project imports
from helocDataset import HELOCDataset
from model import Net
from multiTaskOutputWrapper import MultiTaskOutputWrapper

VALIDATION_SIZE = 2
CSV_FILE = "./heloc_dataset_v1.csv"

def read_csv() -> DataFrame:
    return pd.read_csv(CSV_FILE)

def setup(dataset: DataFrame) -> Tuple[DataLoader, DataLoader]:
    # Read and split data
    dataset_length = len(dataset.values)

    # Instantiate DataSets
    training_split = dataset[:dataset_length//VALIDATION_SIZE]
    HELOC_train = HELOCDataset(training_split)
    train_loader = DataLoader(HELOC_train, num_workers=8, batch_size=64)

    confirmation_split = dataset[dataset_length//VALIDATION_SIZE:]
    test_split = confirmation_split[:len(confirmation_split)//VALIDATION_SIZE]
    HELOC_test = HELOCDataset(test_split)
    test_loader = DataLoader(HELOC_test, num_workers=8, batch_size=64)

    validate_split = confirmation_split[len(confirmation_split)//VALIDATION_SIZE:]
    HELOC_validate = HELOCDataset(validate_split)
    validate_loader = DataLoader(HELOC_validate, num_workers=8, batch_size=64)
    return train_loader, validate_loader, test_loader

def main():
    heloc_dataset = read_csv()
    train_loader, validate_loader, test_loader = setup(heloc_dataset)

    # Configure logging
    api_key = os.environ.get('COMET_API_KEY')
    logger = TensorBoardLogger('lightning_logs')
    if api_key:
        logger = CometLogger(api_key=api_key, project_name='master-jk-pl')
    else:
        print("No Comet-API-key found, defaulting to Tensorboard", flush=True)
        logger.experiment.add_graph(model, train_loader.dataset[0][0].unsqueeze(0)) # Add model graph to Tensorboarf

    # Instantiate model
    nodes_before_split = 64
    input_length = len(train_loader.dataset[0][0])
    net = Net(input_length=input_length, output_length=nodes_before_split)
    model = MultiTaskOutputWrapper(model_core=net, input_length=nodes_before_split, output_length=(1,1))
    trainer = pl.Trainer(max_epochs=150, logger=logger)
    trainer.fit(model, train_loader, validate_loader)
    trainer.test(model,test_loader)


if __name__ == "__main__":
    main()
