import csv
import pandas as pd
import numpy as np
import math

from code.Helpers import getArgs
from code.Sampling import sampling

from models.SingleTaskLearner import SingleTaskLearner

types_url = "./data/heloc"
types_dict_file_name = f"{types_url}/heloc_types_alt.csv"
types_dict_c_file_name = f"{types_url}/heloc_types_c_alt.csv"
heloc_x_file_name = f"{types_url}/heloc_x.csv"
heloc_x_c_file_name = f"{types_url}/heloc_x_c.csv"
heloc_y_file_name = f"{types_url}/heloc_y.csv"

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

def separate_dataset(dataset):
    dataset_length = len(dataset)
    training_dataset = dataset[:math.ceil(dataset_length*0.02)]
    test_dataset = dataset[math.ceil(dataset_length * 0.98):]
    return training_dataset, test_dataset

def main():
    classifier = SingleTaskLearner.load_from_checkpoint("./classifier.ckpt")
    args = getArgs()

    heloc_x = csv_to_list(heloc_x_file_name)
    heloc_x_c = csv_to_list(heloc_x_c_file_name)
    heloc_y = csv_to_list(heloc_y_file_name)

    heloc_x_training, heloc_x_test = separate_dataset(heloc_x)
    heloc_x_c_training, heloc_x_c_test = separate_dataset(heloc_x_c)
    heloc_y_training, heloc_y_test = separate_dataset(heloc_y)
    out = {
        "training": [heloc_y_training, heloc_x_training, heloc_x_c_training],
        "test_counter": [None, heloc_x_test, heloc_x_c_test, heloc_y_test]
    }
    types_dict = csv_to_dict(types_dict_file_name)
    types_dict_c = csv_to_dict(types_dict_c_file_name)
    counterfactuals = sampling(
        settings="",
        types_dict=types_dict,
        types_dict_c=types_dict_c,
        out=out,
        ncounterfactuals=args.ncounterfactuals,
        classifier=classifier,
        n_batches_train=1,
        n_samples_train=1,
        k=2,
        n_input=23,
        degree_active=args.degree_active
    )
    np.savetxt("counterfactuals.txt", counterfactuals, delimiter=",")

if __name__ == "__main__":
    main()
