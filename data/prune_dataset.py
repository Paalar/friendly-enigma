import pandas as pd

from argparse import ArgumentParser
from pandas import DataFrame

CSV_FILE = "./heloc_dataset_v1.csv"


def read_csv() -> DataFrame:
    return pd.read_csv(CSV_FILE)


def save_csv(filename, dataset):
    dataset.to_csv(filename, index=False)


def prune_value(row, predicate, new_value):
    new_row = row.copy()
    for index, value in enumerate(new_row):
        if value == predicate:
            new_row[index] = new_value
    return new_row


def prune_dataset(dataset: DataFrame) -> DataFrame:
    pruned_dataset = dataset.copy()
    columns = pruned_dataset.columns
    columns = columns.drop("RiskPerformance")
    for index, value in enumerate(pruned_dataset.iterrows()):
        if index % 1000 == 0:
            save_csv("heloc_dataset_pruned.csv", pruned_dataset)
        _, content = value
        print(f"Row #{index}")
        # Rows with -9 are all invalid data
        row = content
        if (row[columns] == -9).all():
            pruned_dataset = pruned_dataset.drop(index)
            continue
        # Row, columns with value -7, -8, -9 are also invalid
        if (row[columns] == -9).any():
            row = prune_value(row[columns], -9, 0)
        if (row[columns] == -8).any():
            row = prune_value(row[columns], -8, 0)
        if (row[columns] == -7).any():
            row = prune_value(row[columns], -7, 0)
        pruned_dataset.loc[index, columns] = row
    return pruned_dataset

def remove_rows_from_dataset(dataset: DataFrame, remove_percentage: int) -> DataFrame:
    if remove_percentage == 0:
        return dataset
    shuffled_dataset = dataset.copy().sample(frac=1).reset_index(drop=True)
    dataset_length = len(shuffled_dataset)
    rows_to_be_removed = round(dataset_length * (remove_percentage / 100))
    sliced_dataset = shuffled_dataset[rows_to_be_removed:]
    return sliced_dataset

def main():
    parser = ArgumentParser(description="Pruning dataset")
    parser.add_argument("--prune", choices=["keep", "prune"])
    parser.add_argument("--remove", type=int, default="0")
    parser.add_argument("--name", type=str, default="heloc_dataset_pruned")
    args = parser.parse_args()
    heloc_dataset = read_csv()
    pruned_dataset = prune_dataset(heloc_dataset) if args.prune == "prune" else heloc_dataset
    sliced_dataset = remove_rows_from_dataset(pruned_dataset, args.remove)
    save_csv(f"{args.name}.csv", sliced_dataset)


if __name__ == "__main__":
    main()
