import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import config

cchvaePath = "data/fake"
cchvaeFolder = Path(cchvaePath)
cchvaeFolder.mkdir(exist_ok=True)

dataDir = Path(__file__).parent.absolute()
fake = Path(f"{dataDir}/fake_data_times_100.csv").resolve()
df = pd.read_csv(fake)
records = df.to_numpy()

types_column_names = ["type", "dim", "nclass"]

locked_features = config["cchvae_locked_features_fake"]
free_features = [i for i in range(1, 4) if (i not in locked_features)]

fake_x_c = {index: records[:, value] for index, value in enumerate(locked_features)}
fake_x = {index: records[:, value] for index, value in enumerate(free_features)}

fake_y = records[:, 0]


def save(file, filename):
    df_file = pd.DataFrame(file).astype(float)
    df_file.to_csv(filename, header=False, index=False)


fake_types_c = [["count", 1, ""] for i in range(len(fake_x_c.keys()))]
fake_types = [["real", 1, ""] for i in range(len(fake_x.keys()))]

permutationPathName = "_".join(str(feature) for feature in locked_features)
permutationFolder = Path(f"{cchvaeFolder}/{permutationPathName}")
permutationFolder.mkdir(exist_ok=True)

df_types = pd.DataFrame(fake_types, columns=types_column_names)
df_types_c = pd.DataFrame(fake_types_c, columns=types_column_names)

df_types.to_csv(f"{permutationFolder}/fake_types.csv", index=False)
df_types_c.to_csv(f"{permutationFolder}/fake_types_c.csv", index=False)
save(fake_x_c, f"{permutationFolder}/fake_x_c.csv")
save(fake_x, f"{permutationFolder}/fake_x.csv")
save(fake_y, f"{permutationFolder}/fake_y.csv")
