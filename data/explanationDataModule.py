import pandas as pd

from config import config
from data.helocDataModule import HelocDataModule
from data.explanationDataset import ExplanationDataset
from torch.utils.data import DataLoader


class ExplanationDataModule(HelocDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(self):
        super().prepare_data()
        EXPLANATIONS_CSV_FILE = "data/explanations.csv"
        explanation_data = pd.read_csv(EXPLANATIONS_CSV_FILE)
        explanation_label = explanation_data.iloc[:, 1]
        self.data.insert(0, "explanation set", explanation_label)

    def train_dataloader(self):
        HELOC_train = ExplanationDataset(self.training_split)
        return DataLoader(
            HELOC_train, num_workers=self.workers, batch_size=self.batch_size
        )

    def val_dataloader(self):
        HELOC_validate = ExplanationDataset(self.validate_split)
        return DataLoader(
            HELOC_validate, num_workers=self.workers, batch_size=self.batch_size
        )

    def test_dataloader(self):
        HELOC_test = ExplanationDataset(self.test_split)
        return DataLoader(
            HELOC_test, num_workers=self.workers, batch_size=self.batch_size
        )
