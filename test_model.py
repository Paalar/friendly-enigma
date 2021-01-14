import argparse
import os
import sys
import torch

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from pytorch_lightning import Trainer

from models.singleTaskLearner import SingleTaskLearner
from models.multiTaskLearner import MultiTaskLearner
from data.helocDataModule import HelocDataModule
from data.explanationDataModule import ExplanationDataModule
from utils.print_colors import green_text, red_text, create_cond_print


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

elif os.path.isfile(args.checkpoint):
    checkpoint_file = args.checkpoint

else:
    print("Invalid input for checkpoint. Try again.")
    sys.exit(1)


model_is_mtl = checkpoint_file.split("/")[1].startswith("mtl")
print(f"Recognized model as {'MTL' if model_is_mtl else 'STL' }.")
model = (
    MultiTaskLearner.load_from_checkpoint(checkpoint_file)
    if model_is_mtl
    else SingleTaskLearner.load_from_checkpoint(checkpoint_file)
)

data_module = ExplanationDataModule() if model_is_mtl else HelocDataModule()
data_module.prepare_data()
data_module.setup("test")
trainer = Trainer()
trainer.test(model, datamodule=data_module)

query_model = input("Query model? ")

if query_model:
    data_module.prepare_data()
    data_module.setup("test")
    data_provider = iter(data_module.test_dataloader())

    while query_model == "y" or query_model == "yes":
        input_data, correct_data, correct_explanation = next(data_provider)
        torch.set_printoptions(profile="full")
        toggle_silent = False
        stats_pred_correct = 0
        stats_exp_correct = 0
        for index, batch in enumerate(input_data):
            cond_print = create_cond_print(not toggle_silent)
            predicted_data, predicted_explanation = model(batch)
            cond_print("Predicting on:", batch)
            cond_print("--Prediction--")
            cond_print("Predicted:", predicted_data)
            predicted_correctly = (
                round(predicted_data.item()) == correct_data[index].item()
            )
            if predicted_correctly:
                stats_pred_correct += 1
            print_color_pred = green_text if predicted_correctly else red_text
            cond_print("Correct prediction:", print_color_pred(correct_data[index]))

            cond_print("--Explanation--")
            cond_print("Explained:", predicted_explanation)
            explained_correctly = torch.argmax(predicted_explanation) == torch.argmax(
                correct_explanation[index]
            )

            if explained_correctly:
                stats_exp_correct += 1
            print_color_exp = green_text if explained_correctly else red_text
            cond_print(
                "Correct explanation:", print_color_exp(correct_explanation[index])
            )
            cond_print(
                f"Explained parameter {torch.argmax(predicted_explanation)} as highest."
            )
            cond_print(
                f"Correct has parameter {torch.argmax(correct_explanation[index])} as highest."
            )

            if not toggle_silent:
                cont = input("Continue? [Y: Yes / S: Silently complete batch] ")
                if not (cont == "y" or cont == "yes" or cont == "Y"):
                    if cont == "S" or cont == "s":
                        toggle_silent = True
                    else:
                        break
        print("\n--Batch summary--")
        print(
            f"Predicted correctly: {stats_pred_correct} / {len(correct_data)}. {round((stats_pred_correct/len(correct_data)) * 100, 2)}%"
        )
        print(
            f"Explained correctly: {stats_exp_correct} / {len(correct_explanation)}. {round((stats_exp_correct/len(correct_explanation)) * 100, 2)}%"
        )
        query_model = input("Next batch? ")
        torch.set_printoptions(profile="default")
