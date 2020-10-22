# Package imports
import pandas as pd
import torch

# Subpackage imports
from torch.utils.data import DataLoader
from pandas import DataFrame, Series
from typing import Tuple

# Project imports
from helocDataset import HELOCDataset
from model import Net
from multiTaskOutputWrapper import MultiTaskOutputWrapper

VALIDATION_SIZE = 3
CSV_FILE = "./heloc_dataset_v1.csv"

def read_csv() -> DataFrame:
    return pd.read_csv(CSV_FILE)

def split_predictor(data: DataFrame) -> Tuple[DataFrame, Series]:
    predictions_label = data.pop('RiskPerformance')
    normalized_prediction_labels = [0 if label == "Bad" else 1 for label in predictions_label]
    return data, normalized_prediction_labels

def tensor_predictor(data: Series) -> torch.Tensor:
    #print(data)
    #data = [torch.tensor(value, dtype=torch.float) for value in data]
    data = torch.tensor(data, dtype=torch.float)
    print(data)
    return data

def setup(dataset: DataFrame) -> Tuple[Tuple[DataLoader, Series], Tuple[DataLoader, Series]]:
    # Read and split data
    dataset_length = len(dataset.values)
    dataset_train, train_predictor = split_predictor(dataset[dataset_length//VALIDATION_SIZE:])
    dataset_validate, validate_predictor = split_predictor(dataset[:dataset_length//VALIDATION_SIZE])
    train_predictor = tensor_predictor(train_predictor)
    validate_predictor = tensor_predictor(validate_predictor)

    # Instantiate DataSets
    HELOC_validate = HELOCDataset(dataset_validate)
    HELOC_train = HELOCDataset(dataset_train)
    train_loader = DataLoader(HELOC_train)
    validate_loader = DataLoader(HELOC_validate)
    return (train_loader, train_predictor), (validate_loader, validate_predictor)

def main():
    heloc_dataset = read_csv()
    (train_loader, train_predictor), (validate_loader, validate_predictor) = setup(heloc_dataset)

    # Instantiate model
    nodes_before_split = 50
    input_length = len(train_loader.dataset[0][0])
    net = Net(input_length=input_length, output_length=nodes_before_split)
    model = MultiTaskOutputWrapper(model_core=net, input_length=nodes_before_split, output_length=(1,1))
    model.train(train_loader, train_predictor)
    


if __name__ == "__main__":
    main()
