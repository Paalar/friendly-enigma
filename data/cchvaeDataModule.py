import pandas as pd
from pytorch_lightning import LightningDataModule

from data.helocDataModule import HelocDataModule
from torch.utils.data import DataLoader
from .cchvaeDataset import CchvaeDataset

class CchvaeDataModule(HelocDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_data(self):
        super().prepare_data()
        DELTA_COUNTERFACTUALS_CSV_FILE = "data/delta_counterfactuals.csv"
        counterfactuals_data = pd.read_csv(DELTA_COUNTERFACTUALS_CSV_FILE, header=None)
        squashed_counterfactuals = ["".join(str(row)) for row in counterfactuals_data.values[:,1:]]
        self.data.insert(0, "counterfactual delta", squashed_counterfactuals)

    def train_dataloader(self):
        HELOC_train = CchvaeDataset(self.training_split)
        return DataLoader(
            HELOC_train, num_workers=self.workers, batch_size=self.batch_size
        )

    def val_dataloader(self):
        HELOC_validate = CchvaeDataset(self.validate_split)
        return DataLoader(
            HELOC_validate, num_workers=self.workers, batch_size=self.batch_size
        )

    def test_dataloader(self):
        HELOC_test = CchvaeDataset(self.test_split)
        return DataLoader(
            HELOC_test, num_workers=self.workers, batch_size=self.batch_size
        )
