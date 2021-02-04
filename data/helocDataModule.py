import pandas as pd
import inspect

from pytorch_lightning import LightningDataModule
from .helocDataset import HELOCDataset
from torch.utils.data import DataLoader
from config import config


class HelocDataModule(LightningDataModule):
    def __init__(
        self,
        validation_size: int = 2,
        workers: int = 8,
        batch_size: int = 70,
    ):
        super().__init__()
        self.validation_size = validation_size
        self.workers = config["cpu_workers"] if type(config["cpu_workers"]) is int else workers
        self.batch_size = config["batch_size"] if type(config["batch_size"]) is int else batch_size

    def prepare_data(self):
        CSV_FILE = "data/heloc_dataset_v1.csv"
        self.data = pd.read_csv(CSV_FILE)
        self.row_length = self.data.shape[1] - 1  # Remove predictor
        self.labels = self.data.columns[1:]

    def setup(self, step):
        # step is either 'fit' or 'test', can be used to read only necessary data.
        # Read and split data
        dataset = self.data
        dataset_length = len(dataset.values)
        self.training_split = dataset[: dataset_length // self.validation_size]
        confirmation_split = dataset[dataset_length // self.validation_size :]
        self.test_split = confirmation_split[
            : len(confirmation_split) // self.validation_size
        ]
        self.validate_split = confirmation_split[
            len(confirmation_split) // self.validation_size :
        ]

    def train_dataloader(self):
        HELOC_train = HELOCDataset(self.training_split)
        return DataLoader(
            HELOC_train, num_workers=self.workers, batch_size=self.batch_size
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
