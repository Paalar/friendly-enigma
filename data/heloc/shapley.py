from data.normalized_labels import Labels
import pandas as pd
import torch
from tqdm import tqdm

from captum.attr import ShapleyValueSampling
from pandas.core.frame import DataFrame
from models.singleTaskLearner import SingleTaskLearner

def split_targets_predictors(dataset: DataFrame):
    targets = dataset.iloc[:,0]
    predictors = dataset.iloc[:,1:]
    return targets, predictors

def split_normalized(dataset):
    df = DataFrame(dataset)
    return split_targets_predictors(df)

# heloc_checkpoint = "cchvae/classifier-heloc.ckpt"
heloc_checkpoint = "checkpoints/stl-21-04-21-08-11-33/heloc-epoch=196-loss_validate=0.37-top-validate.ckpt"

heloc = pd.read_csv("data/heloc/heloc_dataset_v1_pruned.csv", index_col=False)
# print(heloc)
heloc_c = pd.read_csv("data/heloc/counterfactuals.csv", header=None, index_col=False)
amplified_heloc_c = pd.read_csv("data/heloc/augmented_counterfactuals.csv", index_col=False)

heloc_t = heloc["RiskPerformance"].map(lambda target: 1 if target == "Good" else 0)
heloc["RiskPerformance"] = heloc_t

classifier = SingleTaskLearner.load_from_checkpoint(heloc_checkpoint)
sv = ShapleyValueSampling(classifier)

heloc_shapley_values = []
labels = Labels(heloc.to_numpy())
targets, predictors = split_normalized(labels.labels)
with tqdm(total=len(targets)) as progress_bar:
    for predictor in predictors.values:
        inp = torch.tensor(predictor, dtype=torch.float32).unsqueeze(0)
        attributions = sv.attribute(inp, target=0)
        heloc_shapley_values.append(attributions)
        progress_bar.update(1)


df = DataFrame(heloc_shapley_values, index=None)
df.to_csv("data/heloc/shapley_heloc_pruned.csv")
