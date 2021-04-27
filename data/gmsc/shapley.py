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

gmsc_checkpoint = "checkpoints/stl_gsmc-21-04-24-10-02-32/heloc-epoch=01-loss_validate=-0.03-top-validate.ckpt"

gmsc = pd.read_csv("data/gmsc/gmsc-training.csv", index_col=False)
# print(heloc)
heloc_c = pd.read_csv("data/gmsc/gmsc_counterfactuals.csv", header=None, index_col=False)
amplified_heloc_c = pd.read_csv("data/gmsc/augmented_counterfactuals.csv", index_col=False)

gmsc_t = gmsc["SeriousDlqin2yrs"]

classifier = SingleTaskLearner.load_from_checkpoint(gmsc_checkpoint)
sv = ShapleyValueSampling(classifier)

heloc_shapley_values = []
labels = Labels(gmsc.to_numpy())
targets, predictors = split_normalized(labels.labels)
with tqdm(total=len(targets)) as progress_bar:
    for predictor in predictors.values:
        inp = torch.tensor(predictor, dtype=torch.float32).unsqueeze(0)
        attributions = sv.attribute(inp, target=0)
        heloc_shapley_values.append(attributions)
        progress_bar.update(1)


df = DataFrame(heloc_shapley_values, index=None)
df.to_csv("data/gmsc/shapley_gmsc_pruned.csv")
