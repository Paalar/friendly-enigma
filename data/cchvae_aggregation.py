import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def get_file(filename):
    dataDir = Path(__file__).parent.absolute()
    return Path(f"{dataDir}/{filename}").resolve()

columns = pd.read_csv(get_file("./heloc_dataset_v1_pruned.csv")).columns
counterfactuals = pd.read_csv(get_file("./counterfactuals.csv"), header=None)
delta_counterfactuals = pd.read_csv(get_file("./delta_counterfactuals.csv"), header=None)
augmented_counterfactuals = []
augmented_delta_counterfactuals = []
progressbar_length = len(delta_counterfactuals.values)


with tqdm(total=progressbar_length) as progressbar:
    for delta_index, delta_row in enumerate(delta_counterfactuals.values):
        delta_row = delta_counterfactuals.values[delta_index]
        counterfactual = list(counterfactuals.values[delta_index])
        zeroed_delta = list(np.zeros(len(counterfactual)))
        for index, delta in enumerate(delta_row):
            delta_to_be_applied = delta_row.copy()
            new_counterfactual = counterfactual.copy()
            if index == 0:
                zeroed_delta.insert(0, delta)
                new_counterfactual.insert(0, delta)
                continue
            if delta == 0:
                continue
            delta_to_be_applied[index] = 0
            applied_zeroed_delta = zeroed_delta.copy()
            applied_zeroed_delta[index] = delta
            new_counterfactual_with_applied_delta = np.add(new_counterfactual, delta_to_be_applied[1:])
            new_counterfactual_with_applied_delta = np.insert(new_counterfactual_with_applied_delta, 0, delta_row[0])
            augmented_counterfactuals.append(new_counterfactual_with_applied_delta)
            augmented_delta_counterfactuals.append(applied_zeroed_delta)
        progressbar.update(1)

df_c = pd.DataFrame(data=augmented_counterfactuals, columns=columns)
df_dc = pd.DataFrame(data=augmented_delta_counterfactuals, columns=columns)
df_c.to_csv("augmented_counterfactuals.csv", index=False)
df_dc.to_csv("augmentted_delta_counterfactuals.csv", index=False, header=None)
