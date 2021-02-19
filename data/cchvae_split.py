import pandas as pd
import numpy as np

heloc = "./heloc_dataset_v1_pruned.csv"

df = pd.read_csv(heloc)
records = df.to_numpy()

heloc_x_c = {"col1": records[:,1], "col2": records[:,2], "col3": records[:,4]}
heloc_x = {"col0": records[:,3]}

for index in range(5, len(records[0])):
    heloc_x[f"row{index}"] = records[:,index]

heloc_y = records[:,0]
heloc_y = [1 if value == "Good" else 0 for value in heloc_y]

def save(file, filename):
    df_file = pd.DataFrame(file).astype(float)
    df_file.to_csv(filename, header=False, index=False)

save(heloc_x_c, "heloc_x_c.csv")
save(heloc_x, "heloc_x.csv")
save(heloc_y, "heloc_y.csv")
