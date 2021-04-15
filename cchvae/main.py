import csv
import pandas as pd
import numpy as np
import math

from cchvae.code.Helpers import getArgs
from cchvae.code.Sampling import sampling

from models.singleTaskLearner import SingleTaskLearner
from config import config

FIRST_HALF = "FIRST"
SECOND_HALF = "SECOND"

dataset = "gmsc" # gmsc or geloc
folder = "2_10"
original_dataset = pd.read_csv("./data/GMSC/gmsc-training.csv").values
types_url = f"./cchvae/data/{dataset}/{folder}"
predictors_length = len(original_dataset[0]) - 1 # minus the target
locked_features = config[f"cchvae_locked_features_{dataset}"]
classifier = SingleTaskLearner.load_from_checkpoint(f"./cchvae/classifier-{dataset}.ckpt")

types_dict_file_name = f"{types_url}/{dataset}_types.csv"
types_dict_c_file_name = f"{types_url}/{dataset}_types_c.csv"
x_file_name = f"{types_url}/{dataset}_x.csv"
x_c_file_name = f"{types_url}/{dataset}_x_c.csv"
y_file_name = f"{types_url}/{dataset}_y.csv"

def csv_to_dict(file_name):
    df = pd.read_csv(file_name)
    records = df.to_dict(orient='records')
    return records        

def csv_to_list(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        data = list(reader)
        numpy_array = np.array(data, ).astype(np.float)
    return numpy_array

def setup_datasets(half):
    x = csv_to_list(x_file_name)
    x_c = csv_to_list(x_c_file_name)
    y = csv_to_list(y_file_name)

    x_training, x_test = separate_dataset(x, half)
    x_c_training, x_c_test = separate_dataset(x_c, half)
    y_training, y_test = separate_dataset(y, half)
    out = {
        "training": [y_training, x_training, x_c_training],
        "test_counter": [None, x_test, x_c_test, y_test]
    }
    return out

def separate_dataset(dataset, half):
    dataset_length = round(len(dataset) * 0.5)
    training_dataset = dataset[dataset_length:] if half == FIRST_HALF else dataset[:dataset_length]
    test_dataset = dataset[:dataset_length] if half == FIRST_HALF else dataset[dataset_length:]
    return training_dataset, test_dataset

def reshape_counterfactuals(counterfactuals):
    shape = counterfactuals.shape
    reshaped_counterfactuals = counterfactuals.reshape(-1, shape[2])
    return reshaped_counterfactuals

def rearrange_counterfactuals(counterfactuals):
    locked_permutation = [feature - 1 for feature in locked_features]
    free_permutation = [i for i in range(predictors_length) if i not in locked_permutation]
    permutation = locked_permutation + free_permutation
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    rearranged_counterfactuals = counterfactuals[:, idx]
    return rearranged_counterfactuals

def calculate_counterfactual_delta(counterfactuals):
    delta = []
    y = original_dataset[:,0]
    predictors = original_dataset[:,1:]
    target_function = get_counterfactual_target_heloc if dataset == "heloc" else get_counterfactual_target_gmsc
    for original, counterfactual, prediction in zip(predictors, counterfactuals, y):
        diff = original - counterfactual
        diff = np.insert(diff, 0, target_function(prediction))
        delta.append(diff)
    return np.asarray(delta)

def get_counterfactual_target_heloc(prediction):
    return "Good" if prediction == "Bad" else "Bad"

def get_counterfactual_target_gmsc(prediction):
    return 0 if prediction else 1

def sample(out, ncounterfactuals):
    args = getArgs()
    types_dict = csv_to_dict(types_dict_file_name)
    types_dict_c = csv_to_dict(types_dict_c_file_name)

    return sampling(
        settings="",
        types_dict=types_dict,
        types_dict_c=types_dict_c,
        out=out,
        ncounterfactuals=ncounterfactuals,
        classifier=classifier,
        n_batches_train=1,
        n_samples_train=1,
        k=1,
        n_input=predictors_length,
        degree_active=args.degree_active
    )

def main():

    out_first = setup_datasets(FIRST_HALF)
    out_second = setup_datasets(SECOND_HALF)

    print(len(out_first["test_counter"][1]))

    print("Creating counterfactuals for the first half of the dataset")
    counterfactuals_first = sample(out_first, len(out_first["test_counter"][1]))
    print("Creating counterfactuals for the second half of the dataset")
    counterfactuals_second = sample(out_second, len(out_second["test_counter"][1]))
    counterfactuals = np.concatenate((counterfactuals_first, counterfactuals_second))

    reshaped_counterfactuals = reshape_counterfactuals(counterfactuals)
    rearranged_counterfactuals = rearrange_counterfactuals(reshaped_counterfactuals)
    delta = calculate_counterfactual_delta(rearranged_counterfactuals)

    delta_format = ["%.0f" for i in range(predictors_length)]
    delta_format.insert(0, "%s")
    np.savetxt(f"{dataset}_delta_counterfactuals.csv", delta, fmt=delta_format, delimiter=",")
    np.savetxt(f"{dataset}_counterfactuals.csv", rearranged_counterfactuals, fmt="%.0f", delimiter=",")

if __name__ == "__main__":
    main()
