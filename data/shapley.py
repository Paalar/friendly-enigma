import pandas as pd
import torch

from captum.attr import ShapleyValues
from pandas.core.frame import DataFrame
from models.singleTaskLearner import SingleTaskLearner

def split_targets_predictors(dataset: DataFrame):
    targets = dataset.iloc[:,0]
    predictors = dataset.iloc[:,1:]
    return targets, predictors

heloc_checkpoint = "cchvae/classifier-heloc.ckpt"

heloc = pd.read_csv("data/heloc/heloc_dataset_v1_pruned.csv", index_col=False)
heloc_c = pd.read_csv("data/heloc/counterfactuals.csv", header=None, index_col=False)
amplified_heloc_c = pd.read_csv("data/heloc/augmented_counterfactuals.csv", index_col=False)

classifier = SingleTaskLearner.load_from_checkpoint(heloc_checkpoint)
sv = ShapleyValues(classifier)

targets, predictors = split_targets_predictors(heloc)

input = torch.tensor(predictors.iloc[0].values, dtype=torch.float32)
target = 1 if targets.iloc[0] == "Good" else 0
# target = torch.tensor([target], dtype=torch.float32)
# torch.tensor(["Good"])
# input = torch.tensor(input)
print(input)

attributions, delta = sv.attribute(input, target=target)

print(attributions)
print(delta)
