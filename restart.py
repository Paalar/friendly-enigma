import comet_ml
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

from config import config
from utils import dashboard
from models.multiTaskLearner import MultiTaskLearner
from argparse import ArgumentParser

from data.gmscDataModule import GMSCDataModule
from data.stlGmscDataModule import STLGMSCDataModule
from data.stlFakeDataModule import STLFakeDataModule
from data.mtlFakeDataModule import MTLFakeDataModule

models = {
    # "stl": STLRunner,
    # "mtl": MTLRunner,
    # "cchvae": Counterfactual_Runner,
    "gmsc": GMSCDataModule,
    "stl_gmsc": STLGMSCDataModule,
    "stl_fake": STLFakeDataModule,
    "fake": MTLFakeDataModule
}

parser = ArgumentParser(description="A multitask learner-restarter")
parser.add_argument("model_type", choices=models.keys(), help="")
args = parser.parse_args()

def recursivelySelectFile(file_or_folder="checkpoints"):
    if os.path.isfile(file_or_folder):
        return file_or_folder
    ls = os.listdir(file_or_folder)
    ls.sort(reverse=True)
    ls_completer = WordCompleter(ls)
    file_or_folder_input = prompt(
        "Pick file or folder, press tab to see options: ", completer=ls_completer
    )
    new_file_or_folder = file_or_folder + "/" + file_or_folder_input
    return recursivelySelectFile(new_file_or_folder)

def create_checkpoint_callbacks(prefix):
    period_callback = ModelCheckpoint(
        period=(config['stl_epochs'] if prefix == "stl" else config["mtl_epochs"])/10,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename=prefix+"-{epoch:02d}-{loss_validate:.2f}-period",
        save_last=True
    )
    loss_validate_callback = ModelCheckpoint(
        monitor="loss_validate",
        save_top_k=3,
        dirpath=f"./checkpoints/{prefix}-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}",
        filename=prefix+"-{epoch:02d}-{loss_validate:.2f}-top-validate",
    )
    return period_callback, loss_validate_callback

logger = dashboard.create_logger()
checkpoint = recursivelySelectFile()
module = models.get(args.model_type)()
model = MultiTaskLearner.load_from_checkpoint(checkpoint)
trainer = Trainer(resume_from_checkpoint=checkpoint, logger=logger, callbacks=[*create_checkpoint_callbacks("gmsc")],)
trainer.fit(model, module)
trainer.test(model, datamodule=module)
