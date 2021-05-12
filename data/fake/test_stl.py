import pandas as pd
import torch
import torchmetrics

from pathlib import Path

from utils.file import recursivelySelectFile
from models.singleTaskLearner import SingleTaskLearner

VAL = 100

def get_sum_with_max_contribution_per_value(tensor: torch.Tensor):
    values = tensor.tolist()
    target_val = 0
    for value in values:
        positive_capped = value if value < 0.5 else 0.5
        negative_capped = positive_capped if positive_capped > -0.5 else -0.5
        target_val += negative_capped
    return target_val

dataDir = Path(__file__).parent.absolute()
counterfactuals = pd.read_csv(f"{dataDir}/fake_counterfactuals.csv", header=None)
counterfactuals = counterfactuals.reindex(columns=[0, 1, 4, 5, 2, 3])
real = pd.read_csv(f"{dataDir}/fake_data.csv")
real2 = pd.read_csv(f"{dataDir}/fake_data_2.csv").iloc[:,1:]
real_t = real.iloc[:,0]
# checkpoint = recursivelySelectFile()
checkpoint = "checkpoints/stl_fake-21-04-28-14-50-40/heloc-epoch=99-loss_validate=0.06-top-validate.ckpt"
# checkpoint = "checkpoints/stl_fake-21-05-04-13-40-34/last.ckpt"
# checkpoint = "checkpoints/stl_fake-21-05-04-14-06-09/last.ckpt"
# checkpoint = "checkpoints/stl_fake-21-05-05-16-58-36/heloc-epoch=35-loss_validate=0.00-top-validate.ckpt"
checkpoint = "checkpoints/stl_fake-21-05-05-19-03-09/last.ckpt"

model = SingleTaskLearner.load_from_checkpoint(checkpoint)
print(model.learning_rate)
print(model.named_parameters)
print(model.test_dataloader())

c = 0
correct = []
false = []

l = model.predict(torch.tensor([50, -9, -6, 51, 0, 0], dtype=torch.float))
print(l)
l = model.predict(torch.tensor([100, 100, 100, 100, 0, 0], dtype=torch.float))
print(l)
l = model.predict(torch.tensor([42.71441,16.473236,42.958824,-11.516689, 0, 0], dtype=torch.float))
print(l)
l = model.predict(torch.tensor([420 ,160, 420, 11, 0, 0], dtype=torch.float))
print(l)

targets = torch.tensor(
    [
        1 if get_sum_with_max_contribution_per_value(tensor) > (0.5 * VAL) else 0
        for tensor in counterfactuals.values
    ]
)

accs = []

for index, row in enumerate(real.iloc[:,1:].values):
    target = targets[index]
    pred = model.predict(row)
    accs.append(torchmetrics.functional.accuracy(pred, target))
    print(f"{pred} - {real_t[index]}")
    # print("\n")
    item = pred.item()
    if item == 1 and real_t[index] == 0:
        # print("1-0")
        correct.append(item)
    elif item == 0 and real_t[index] == 1:
        # print("0-1")
        correct.append(item)
    else:
        # print("1-1 or 0-0")
        false.append(item)
print("Correctly predicted counterfactuals", len(correct))
print("Falsely predicted counterfactuals", len(false))

print("Acc", sum(accs)/len(accs))


# print(torch.bincount(torch.logical_xor(targets, torch.tensor(real_t.values)).long()))
xor = torch.logical_xor(targets, torch.tensor(real_t.values)).long()
zero = 0
one = 0
for i in xor.tolist():
    if i == 0:
        zero += 1
    if i == 1:
        one += 1

print("False generated counterfactuals", zero)
print("Correct generated counterfactuals", one)
print(one / (one + zero))
