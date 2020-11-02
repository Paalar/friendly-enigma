import argparse
import torch
import shap
import pandas
import edc
import numpy as np

from pytorch_lightning import Trainer
from edc.sedc_agnostic.sedc_algorithm import SEDC_Explainer 

from multiTaskOutputWrapper import MultiTaskOutputWrapper
from data.helocDataModule import HelocDataModule
from model import Net
# Load the model
parser = argparse.ArgumentParser(description="Load and evaluate a saved model")
parser.add_argument("checkpoint", type=str, help="File path to load")
args = parser.parse_args(["checkpoints/2020-10-29T13:23:40.072673/heloc-epoch=12-loss_validate=-0.25.ckpt"])
model = MultiTaskOutputWrapper.load_from_checkpoint(args.checkpoint)

# Test the model
data_module = HelocDataModule()
data_module.prepare_data()
data_module.setup("train")
#trainer = Trainer()
#trainer.test(model, datamodule=data_module)

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
    feature_names = np.array(data_module.labels),
    threshold_classifier = np.percentile(predictions, 75),
    classifier_fn = classifier_fn,
    silent = True
)

explanations = []
for loader in loaders:
    for batch in iter(loader):
        values, _ = batch
        for value in values:
            explanation = sedc_explainer.explanation(value)
            explanations.append(explanation)

    
df = pandas.DataFrame(explanations)
df.to_csv("data/explanations.csv")
