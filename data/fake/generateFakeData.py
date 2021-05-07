import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
import numpy as np
import os

np.random.seed(123)
os.environ["PYTHONHASHSEED"] = str(123)
torch.manual_seed(123)

VAL = 100


def get_sum_with_max_contribution_per_value(tensor: torch.Tensor):
    values = tensor.tolist()
    target_val = 0
    for value in values:
        positive_capped = value if value < 0.5 else 0.5
        negative_capped = positive_capped if positive_capped > -0.5 else -0.5
        target_val += negative_capped
    return target_val


epsilond = 0


def get_sum_with_max_contribution_per_value_with_epsilon(tensor: torch.Tensor):
    global epsilond
    values = tensor.tolist()
    target_val = 0
    epsilon = np.random.random()
    for value in values:
        positive_capped = value if value < 0.5 else 0.5
        negative_capped = positive_capped if positive_capped > -0.5 else -0.5
        target_val += negative_capped
    target_val = target_val * -1 if epsilon < 0.1 else target_val
    if epsilon < 0.1:
        epsilond += 1
    return target_val


initial_random_data = torch.round(torch.rand(50000, 4) * VAL)
initial_sum_to_1 = initial_random_data - (0.5 * VAL)

# A positive target is one where the sum is over 0.5, and a single value cannot contribute more than 0.5 towards the sum.
targets = torch.tensor(
    [
        1 if get_sum_with_max_contribution_per_value_with_epsilon(tensor) > 0.5 else 0
        for tensor in initial_sum_to_1
    ]
)
print("Generated fake data, amount of true targets: ", torch.count_nonzero(targets))
print("Samples turned by noise: ", epsilond)

# Create names for columns and insert into csv.
column_names = [f"X_{num}" for num in range(initial_sum_to_1.size()[1])]
df = pd.DataFrame(initial_sum_to_1.numpy(), columns=column_names)
df.insert(0, "Targets", targets)

# Insert locked features
df.insert(len(df.values[0]), "X_4", torch.zeros(50000, 1).numpy())
df.insert(len(df.values[0]), "X_5", torch.zeros(50000, 1).numpy())

# Save CSV
dataDir = Path(__file__).parent.absolute()
df.to_csv(f"{dataDir}/fake_data_3.csv", index=False)
