import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
import math
from argparse import ArgumentParser

parser = ArgumentParser(description="Fake data generation for Friendly Enigma")
parser.add_argument("noise", type=int, nargs="?", default=0, help="Noise threshold in percentage")

args = parser.parse_args()

np.random.seed(123)
os.environ["PYTHONHASHSEED"] = str(123)
torch.manual_seed(123)

VAL = 100
max_val = 20


def get_sum_with_max_contribution_per_value(tensor: torch.Tensor):
    values = tensor.tolist()
    target_val = 0
    for value in values:
        positive_capped = value if value < max_val else max_val
        negative_capped = positive_capped if positive_capped > -max_val else -max_val
        target_val += negative_capped
    return target_val


epsilond = 0


def get_sum_with_max_contribution_per_value_with_epsilon(tensor: torch.Tensor):
    global epsilond
    global args
    epsilon_threshold = args.noise / 100
    values = tensor.tolist()
    target_val = 0
    epsilon = np.random.random()
    for value in values:
        positive_capped = value if value < max_val else max_val
        negative_capped = positive_capped if positive_capped > -max_val else -max_val
        target_val += negative_capped
    target_val = target_val * -1 if epsilon < epsilon_threshold else target_val
    if epsilon <= epsilon_threshold:
        epsilond += 1
    return target_val


initial_random_data = torch.round(torch.rand(50000, 4) * VAL)
initial_sum_to_1 = initial_random_data - (0.5 * VAL)

# A positive target is one where the sum is over 0.5, and a single value cannot contribute more than 0.5 towards the sum.
test_split = math.floor(0.25 * len(initial_sum_to_1))
test_set = initial_sum_to_1[:test_split, :]
training_set = initial_sum_to_1[test_split:, :]
test_targets = torch.tensor(
    [
        1 if get_sum_with_max_contribution_per_value(tensor) > max_val else 0
        for tensor in test_set
    ]
)
training_targets = torch.tensor(
    [
        1
        if get_sum_with_max_contribution_per_value_with_epsilon(tensor) > max_val
        else 0
        for tensor in training_set
    ]
)
targets = torch.tensor([*training_targets, *test_targets])
print("Generated fake data, amount of true targets: ", torch.count_nonzero(targets))
print("Samples turned by noise: ", epsilond)

# Create names for columns and insert into csv.
column_names = [f"X_{num}" for num in range(initial_sum_to_1.size()[1])]
df = pd.DataFrame([*training_set.numpy(), *test_set.numpy()], columns=column_names)
df.insert(0, "Targets", targets)

# Insert locked features
# df.insert(len(df.values[0]), "X_4", torch.zeros(50000, 1).numpy())
# df.insert(len(df.values[0]), "X_5", torch.zeros(50000, 1).numpy())

# Save CSV
dataDir = Path(__file__).parent.absolute()
df.to_csv(f"{dataDir}/fake_data_epsilond.csv", index=False)
