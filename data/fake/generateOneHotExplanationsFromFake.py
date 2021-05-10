import pandas as pd
import torch
from pathlib import Path
import numpy as np

data_dir = Path(__file__).parent.absolute()
fake_data = pd.read_csv(f"{data_dir}/fake_data_epsilond.csv")
targets = fake_data.iloc[:, 0]
data = fake_data.iloc[:, 1:]
mins = data.idxmin(axis=1)
maxs = data.idxmax(axis=1)
column_names = list(data.columns)
explanations = [
    column_names.index(mins[index]) if target == 0 else column_names.index(maxs[index])
    for index, target in enumerate(targets)
]
one_hot = np.identity(6)[explanations]
pd.DataFrame(one_hot).to_csv(f"{data_dir}/manual_explanations.csv", index=False)
