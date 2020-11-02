import pandas as pd

from helocDataModule import HelocDataModule

class ExplanationDataModule(HelocDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(self):
        super().prepare_data()
        EXPLANATIONS_CSV_FILE = "./explanations.csv"
        explanation_data = pd.read_csv(EXPLANATIONS_CSV_FILE)
        explanation_label = explanation_data.iloc[:,1]
        explanation_label.join(self.data)
        self.data = explanation_label

