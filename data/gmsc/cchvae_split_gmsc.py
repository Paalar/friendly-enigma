import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import config

cchvaePath = "data/GMSC/cchvae"
cchvaeFolder = Path(cchvaePath)
cchvaeFolder.mkdir(exist_ok=True)

dataDir = Path(__file__).parent.absolute()
gmsc = Path(f"{dataDir}/gmsc-training.csv").resolve()
df = pd.read_csv(gmsc)
records = df.to_numpy()

types_column_names = ["type", "dim", "nclass"]

locked_features = config["cchvae_locked_features_gmsc"]
free_features = [i for i in range(1, 11) if (i not in locked_features)]

gmsc_x_c = {index: records[:, value] for index, value in enumerate(locked_features)}
gmsc_x = {index: records[:, value] for index, value in enumerate(free_features)}

gmsc_y = records[:, 0]


def save(file, filename):
    df_file = pd.DataFrame(file).astype(float)
    df_file.to_csv(filename, header=False, index=False)


gmsc_types_c = [["count", 1, ""] for i in range(len(gmsc_x_c.keys()))]
gmsc_types = [["pos", 1, ""], ["count", 1, ""], ["pos", 1, ""], ["pos", 1, ""], ["count", 1, ""], ["count", 1, ""], ["count", 1, ""], ["count", 1, ""]]

permutationPathName = "_".join(str(feature) for feature in locked_features)
permutationFolder = Path(f"{cchvaeFolder}/{permutationPathName}")
permutationFolder.mkdir(exist_ok=True)

df_types = pd.DataFrame(gmsc_types, columns=types_column_names)
df_types_c = pd.DataFrame(gmsc_types_c, columns=types_column_names)

df_types.to_csv(f"{permutationFolder}/gmsc_types.csv", index=False)
df_types_c.to_csv(f"{permutationFolder}/gmsc_types_c.csv", index=False)
save(gmsc_x_c, f"{permutationFolder}/gmsc_x_c.csv")
save(gmsc_x, f"{permutationFolder}/gmsc_x.csv")
save(gmsc_y, f"{permutationFolder}/gmsc_y.csv")
