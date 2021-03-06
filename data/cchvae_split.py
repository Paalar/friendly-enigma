import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import config

cchvaePath = "data/cchvae"
cchvaeFolder = Path(cchvaePath)
cchvaeFolder.mkdir(exist_ok=True)

dataDir = Path(__file__).parent.absolute()
heloc = Path(f"{dataDir}/heloc/heloc_dataset_v1_pruned.csv").resolve()
df = pd.read_csv(heloc)
records = df.to_numpy()

types_column_names = ["type", "dim", "nclass"]

locked_features = config["cchvae_locked_features_heloc"]
free_features = [i for i in range(1, 24) if (i not in locked_features)]

heloc_x_c = {index: records[:, value] for index, value in enumerate(locked_features)}
heloc_x = {index: records[:, value] for index, value in enumerate(free_features)}

heloc_y = records[:, 0]
heloc_y = [1 if value == "Good" else 0 for value in heloc_y]


def save(file, filename):
    df_file = pd.DataFrame(file).astype(float)
    df_file.to_csv(filename, header=False, index=False)


heloc_types_c = [["count", 1, ""] for i in range(len(heloc_x_c.keys()))]
heloc_types = [["real", 1, ""] for i in range(len(heloc_x.keys()))]

permutationPathName = "_".join(str(feature) for feature in locked_features)
permutationFolder = Path(f"{cchvaeFolder}/{permutationPathName}")
permutationFolder.mkdir(exist_ok=True)

df_types = pd.DataFrame(heloc_types, columns=types_column_names)
df_types_c = pd.DataFrame(heloc_types_c, columns=types_column_names)

df_types.to_csv(f"{permutationFolder}/heloc_types.csv", index=False)
df_types_c.to_csv(f"{permutationFolder}/heloc_types_c.csv", index=False)
save(heloc_x_c, f"{permutationFolder}/heloc_x_c.csv")
save(heloc_x, f"{permutationFolder}/heloc_x.csv")
save(heloc_y, f"{permutationFolder}/heloc_y.csv")
