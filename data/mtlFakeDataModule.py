import pandas as pd
import math

from pytorch_lightning import LightningDataModule
from torch.utils import data
from .mtlFakeDataset import MTLFakeDataset
from torch.utils.data import DataLoader
from config import config


class MTLFakeDataModule(LightningDataModule):
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
        self.workers = (
            config["cpu_workers"] if type(config["cpu_workers"]) is int else workers
        )
        self.batch_size = (
            config["batch_size"] if type(config["batch_size"]) is int else batch_size
        )

    def prepare_data(self):
        CSV_FILE = (
            "data/fake/fake_data.csv"  # "data/fake/augmented_counterfactuals_5_6.csv"
        )
        self.data = pd.read_csv(CSV_FILE)
        self.row_length = self.data.shape[1] - 1  # Remove predictor
        self.labels = self.data.columns[1:]
        print("Length of labels: ", len(self.labels))
        DELTA_COUNTERFACTUALS_CSV_FILE = "data/fake/manual_explanations.csv"  # "data/fake/augmented_delta_counterfactuals_5_6.csv"
        counterfactuals_data = pd.read_csv(DELTA_COUNTERFACTUALS_CSV_FILE)
        squashed_counterfactuals = [
            "".join(str(row)) for row in counterfactuals_data.values[:, 1:]
        ]
        print("Length of squashed: ", len(squashed_counterfactuals))
        self.data.insert(0, "counterfactual delta", squashed_counterfactuals)

    def setup(self, step):
        # step is either 'fit' or 'test', can be used to read only necessary data.
        # Read and split data
        dataset = self.data
        dataset_length = len(dataset.values)
        training_split_length = math.floor(dataset_length * self.training_size)
        test_split_length = training_split_length + math.floor(
            dataset_length * self.test_size
        )
        self.training_split = dataset[:training_split_length]
        self.test_split = dataset[training_split_length:test_split_length]
        self.validate_split = dataset[test_split_length:]

    def train_dataloader(self):
        GMSC_train = MTLFakeDataset(self.training_split)
        return DataLoader(
            GMSC_train,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        GMSC_validate = MTLFakeDataset(self.validate_split)
        return DataLoader(
            GMSC_validate, num_workers=self.workers, batch_size=self.batch_size
        )

    def test_dataloader(self):
        GMSC_test = MTLFakeDataset(self.test_split)
        return DataLoader(
            GMSC_test, num_workers=self.workers, batch_size=self.batch_size
        )
