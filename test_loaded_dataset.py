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

batch = next(iter(data_module.train_dataloader()))
values, preds = batch
value_idx = values[0]
print(value_idx)
print(value_idx.dim())
# Explain: The rest of the fucking master
sedc_explainer = SEDC_Explainer(
    feature_names = np.array(data_module.labels),
    threshold_classifier = np.percentile(value_idx.unsqueeze(0), 75),
    classifier_fn = classifier_fn,
)
sedc_explainer.explanation(value_idx.unsqueeze(0))



"""
explainer = shap.DeepExplainer(model, values)
next_batch = next(iter(data_module.test_dataloader()))
explain_values, _ = next_batch
explanation = explainer.shap_values(explain_values[0].unsqueeze(0))
#pandas.set_option("display.max_rows", None, "display.max_columns", None)
print(pandas.DataFrame(explanation, columns=data_module.labels))
"""