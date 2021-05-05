import pandas as pd
import math
from pytorch_lightning import LightningDataModule


def get_partial_data_module(module, training_size_percentage):
    class PartialDataModule(module):
        def __init__(
            self,
            workers: int = 8,
            batch_size: int = 70,
        ):
            super().__init__(workers=workers, batch_size=batch_size)
            self.training_size = training_size_percentage / 100
            self.test_size = 25 / 100

        def setup(self, step):
            dataset = self.data
            dataset_length = len(dataset.values)
            split = math.floor(dataset_length * (1 - self.test_size))
            training_dataset = dataset[:split]
            validation_dataset = dataset[split:]
            training_split_length = math.floor(
                len(training_dataset.values) * self.training_size
            )
            self.training_split = training_dataset[:training_split_length]
            test_split = math.floor(0.6 * len(validation_dataset))
            self.test_split = validation_dataset[:test_split]
            self.validate_split = validation_dataset[test_split:]

    return PartialDataModule
