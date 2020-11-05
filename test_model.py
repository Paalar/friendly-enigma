import argparse
import os
import sys

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from pytorch_lightning import Trainer

from models.singleTaskLearner import SingleTaskLearner
from data.helocDataModule import HelocDataModule


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


parser = argparse.ArgumentParser(description="Load and evaluate a saved model")
parser.add_argument("--checkpoint", type=str, help="File path to load")
args = parser.parse_args()
if not args.checkpoint:
    checkpoint_file = recursivelySelectFile()

elif os.path.isdir(args.checkpoint):
    print("Checkpoint is a directory. Which file/directory did you mean?")
    checkpoint_file = recursivelySelectFile(args.checkpoint)

else:
    print("Invalid input for checkpoint. Try again.")
    sys.exit(1)


model = SingleTaskLearner.load_from_checkpoint(checkpoint_file)

data_module = HelocDataModule()
data_module.prepare_data()
data_module.setup("test")
trainer = Trainer()
trainer.test(model, datamodule=data_module)
