import pandas as pd
import math

from pytorch_lightning import LightningDataModule
from torch.utils import data
from .helocDataset import HELOCDataset
from torch.utils.data import DataLoader
from config import config


class HelocDataModule(LightningDataModule):
    def __init__(
        self,
        workers: int = 8,
        batch_size: int = 70,
    ):
        super().__init__()
        self.training_size = config["training_size_percentage"] / 100
        self.test_size = config["test_size_percentage"] / 100
        self.validation_size = 1 - self.training_size - self.test_size
        self.workers = (
            config["cpu_workers"] if type(config["cpu_workers"]) is int else workers
        )
        self.batch_size = (
            config["batch_size"] if type(config["batch_size"]) is int else batch_size
        )

    def prepare_data(self):
        # CSV_FILE = "data/counterfactualsT.csv"
        CSV_FILE = "data/augmented_counterfactuals.csv"
        # CSV_FILE = "data/heloc_dataset_v1_pruned.csv"
        self.data = pd.read_csv(CSV_FILE)
        self.row_length = self.data.shape[1] - 1  # Remove predictor
        self.labels = self.data.columns[1:]

    def setup(self, step):
        # step is either 'fit' or 'test', can be used to read only necessary data.
        # Read and split data
        dataset = self.data
        dataset_length = len(dataset.values)
        training_split_length = math.floor(dataset_length * self.training_size)
        test_split_length = training_split_length + math.floor(dataset_length * self.test_size)
        self.training_split = dataset[:training_split_length]
        self.test_split = dataset[training_split_length:test_split_length]
        self.validate_split = dataset[test_split_length:]

    def train_dataloader(self):
        HELOC_train = HELOCDataset(self.training_split)
        return DataLoader(
            HELOC_train,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        HELOC_validate = HELOCDataset(self.validate_split)
        return DataLoader(
            HELOC_validate, num_workers=self.workers, batch_size=self.batch_size
        )

    def test_dataloader(self):
        HELOC_test = HELOCDataset(self.test_split)
        return DataLoader(
            HELOC_test, num_workers=self.workers, batch_size=self.batch_size
        )
