import csv
import pandas as pd
import numpy as np
import math

from cchvae.code.Helpers import getArgs
from cchvae.code.Sampling import sampling

from models.singleTaskLearner import SingleTaskLearner
from config import config

types_url = "./cchvae/data/heloc"
types_dict_file_name = f"{types_url}/heloc_types.csv"
types_dict_c_file_name = f"{types_url}/heloc_types_c.csv"
heloc_x_file_name = f"{types_url}/heloc_x.csv"
heloc_x_c_file_name = f"{types_url}/heloc_x_c.csv"
heloc_y_file_name = f"{types_url}/heloc_y.csv"

FIRST_HALF = "FIRST"
SECOND_HALF = "SECOND"

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
    heloc_x = csv_to_list(heloc_x_file_name)
    heloc_x_c = csv_to_list(heloc_x_c_file_name)
    heloc_y = csv_to_list(heloc_y_file_name)

    heloc_x_training, heloc_x_test = separate_dataset(heloc_x, half)
    heloc_x_c_training, heloc_x_c_test = separate_dataset(heloc_x_c, half)
    heloc_y_training, heloc_y_test = separate_dataset(heloc_y, half)
    out = {
        "training": [heloc_y_training, heloc_x_training, heloc_x_c_training],
        "test_counter": [None, heloc_x_test, heloc_x_c_test, heloc_y_test]
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
    # Change location of 3 and 2
    locked_features = config["cchvae_locked_features"]
    locked_permutation = [feature - 1 for feature in locked_features]
    free_permutation = [i for i in range(23) if i not in locked_permutation]
    permutation = locked_permutation + free_permutation
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    rearranged_counterfactuals = counterfactuals[:, idx]
    return rearranged_counterfactuals

def calculate_counterfactual_delta(counterfactuals):
    original_dataset = pd.read_csv("./data/heloc_dataset_v1_pruned.csv").values
    delta = []
    y = original_dataset[:,0]
    predictors = original_dataset[:,1:]
    for original, counterfactual, prediction in zip(predictors, counterfactuals, y):
        diff = original - counterfactual
        diff = np.insert(diff, 0, "Good" if prediction == "Bad" else "Bad")
        delta.append(diff)
    return np.asarray(delta)

def sample(out, ncounterfactuals):
    args = getArgs()
    classifier = SingleTaskLearner.load_from_checkpoint("./cchvae/classifier.ckpt")
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
        n_input=23,
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

    delta_format = ["%.0f" for i in range(23)]
    delta_format.insert(0, "%s")
    np.savetxt("delta_counterfactuals.csv", delta, fmt=delta_format, delimiter=",")
    np.savetxt("counterfactuals.csv", rearranged_counterfactuals, fmt="%.0f", delimiter=",")

if __name__ == "__main__":
    main()
