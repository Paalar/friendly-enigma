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

def remove_rows_from_dataset(dataset: DataFrame, rows_to_skip: int) -> DataFrame:
    if rows_to_skip == 0:
        return dataset
    sliced_dataset = dataset.copy()
    counter = 1
    for index, value in enumerate(sliced_dataset.iterrows()):
        counter = 1 if rows_to_skip + 1 == counter else counter
        if rows_to_skip == counter:
            sliced_dataset = sliced_dataset.drop(index)
        counter += 1
    return sliced_dataset

def main():
    parser = ArgumentParser(description="Pruning dataset")
    parser.add_argument("--prune", choices=["keep", "prune"])
    parser.add_argument("--skip", type=int, default="0")
    parser.add_argument("--name", type=str, default="heloc_dataset_pruned")
    args = parser.parse_args()
    heloc_dataset = read_csv()
    pruned_dataset = prune_dataset(heloc_dataset) if args.prune == "prune" else heloc_dataset
    sliced_dataset = remove_rows_from_dataset(pruned_dataset, args.skip)
    save_csv(f"{args.name}.csv", sliced_dataset)


if __name__ == "__main__":
    main()
