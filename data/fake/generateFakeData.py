import torch
import pandas as pd
from pathlib import Path

np.random.seed(123)
os.environ["PYTHONHASHSEED"] = str(123)
torch.manual_seed(123)

dataDir = Path(__file__).parent.absolute()
initial_random_data = torch.rand(50000, 4)
initial_sum_to_1 = initial_random_data - 0.5
sums = torch.sum(initial_sum_to_1, axis=1)
targets = torch.tensor([1 if target > 0.5 else 0 for target in sums])
column_names = [f"X_{num}" for num in range(initial_sum_to_1.size()[1])]
df = pd.DataFrame(initial_sum_to_1.numpy(), columns=column_names)
df.insert(0, "Targets", targets)
df.insert(len(df.values[0]), "X_4", torch.zeros(50000, 1).numpy())
df.insert(len(df.values[0]), "X_5", torch.zeros(50000, 1).numpy())
df.to_csv(f"{dataDir}/fake_data.csv", index=False)
