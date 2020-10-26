# Package imports
import pandas as pd
import pytorch_lightning as pl

# Subpackage
from torch.utils.data import DataLoader
from pandas import DataFrame
from typing import Tuple
from pytorch_lightning.loggers import TensorBoardLogger

# Project imports
from helocDataset import HELOCDataset
from model import Net
from multiTaskOutputWrapper import MultiTaskOutputWrapper

VALIDATION_SIZE = 3
CSV_FILE = "./heloc_dataset_v1.csv"

def read_csv() -> DataFrame:
    return pd.read_csv(CSV_FILE)

def setup(dataset: DataFrame) -> Tuple[DataLoader, DataLoader]:
    # Read and split data
    dataset_length = len(dataset.values)

    # Instantiate DataSets
    HELOC_validate = HELOCDataset(dataset[:dataset_length//VALIDATION_SIZE])
    HELOC_train = HELOCDataset(dataset[dataset_length//VALIDATION_SIZE:])
    train_loader = DataLoader(HELOC_train, num_workers=4, batch_size=512, shuffle=True)
    validate_loader = DataLoader(HELOC_validate, num_workers=4, batch_size=512)
    return train_loader, validate_loader

def main():
    heloc_dataset = read_csv()
    train_loader, validate_loader = setup(heloc_dataset)
    logger = TensorBoardLogger('lightning_logs')
    # Instantiate model
    nodes_before_split = 50
    input_length = len(train_loader.dataset[0][0])
    net = Net(input_length=input_length, output_length=nodes_before_split)
    model = MultiTaskOutputWrapper(model_core=net, input_length=nodes_before_split, output_length=(1,1))
    logger.experiment.add_graph(model, train_loader.dataset[0][0].unsqueeze(0))
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader)
    trainer.test(model, validate_loader)
    # trainer.run_evaluation()


if __name__ == "__main__":
    main()
