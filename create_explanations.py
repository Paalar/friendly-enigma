import argparse
import torch
import pandas
import edc
import numpy as np

from pytorch_lightning import Trainer
from edc.sedc_agnostic.sedc_algorithm import SEDC_Explainer
from tqdm import tqdm

from models.singleTaskLearner import SingleTaskLearner
from data.helocDataModule import HelocDataModule
from models.multiTaskLearner import Net
from config import config
from math import ceil

# Load the model
parser = argparse.ArgumentParser(
    description="Generate counterfactuals as .csv for a given model"
)
parser.add_argument("checkpoint", type=str, help="File path to load")
parser.add_argument(
    "--merge",
    type=bool,
    default=False,
    help="Create a new merged dataset with explanation column",
)
args = parser.parse_args()
model = SingleTaskLearner.load_from_checkpoint(args.checkpoint)

# Test the model
data_module = HelocDataModule()
data_module.prepare_data()
data_module.setup("train")


def classifier_fn(instance):
    tensor = torch.from_numpy(instance.toarray())
    prediction = model(tensor)
    return prediction.detach().numpy()


train_dataloader = data_module.train_dataloader()
validate_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

loaders = [train_dataloader, validate_dataloader, test_dataloader]

predictions = []
for loader in loaders:
    for batch in iter(loader):
        values, preds = batch
        value_idx = values

        prediction = model(value_idx).detach()
        prediction = torch.flatten(prediction)
        predictions += prediction

# Explain: The rest of the fucking master
sedc_explainer = SEDC_Explainer(
    feature_names=np.array(data_module.labels),
    threshold_classifier=np.percentile(predictions, 75),
    classifier_fn=classifier_fn,
    silent=True,
    max_explained=1,
)


# TQDM-preparations
dataset_length = sum([len(loader.dataset) for loader in loaders])
batch_size = config["batch_size"]
progressbar_length = ceil(dataset_length / batch_size)

explanations = []
with tqdm(total=progressbar_length) as progressbar:
    for loader in loaders:
        for batch in iter(loader):
            progressbar.update(1)
            values, _ = batch
            for value in values:
                explanation = sedc_explainer.explanation(value)
                explanations.append(explanation)


df = pandas.DataFrame(explanations)
if not args.merge:
    df.to_csv("data/explanations.csv")
else:
    explanation_label = df.iloc[:, 0].transform(lambda x: list(np.array(x).flatten()))
    heloc = pandas.read_csv("data/heloc_dataset_v1.csv")
    heloc.insert(1, "explanation_label", explanation_label)
    heloc.to_csv("data/heloc_with_explanations.csv", index=False)
