import pandas as pd

from pandas import DataFrame

CSV_FILE = "./heloc_dataset_v1.csv"

def read_csv() -> DataFrame:
    return pd.read_csv(CSV_FILE)

def save_csv(filename, dataset):
    dataset.to_csv(filename, index=False)

def prune_value(row, predicate, new_value):
    new_row = row.copy()
    for index, value in enumerate(new_row):
        if (value == predicate):
            new_row[index] = new_value
    return new_row

def prune_dataset(dataset: DataFrame) -> DataFrame:
    pruned_dataset = dataset.copy()
    columns = pruned_dataset.columns
    columns = columns.drop('RiskPerformance')
    for index, value in enumerate(pruned_dataset.iterrows()):
        if index % 1000 == 0:
            save_csv("heloc_dataset_pruned.csv", pruned_dataset)
        _, content = value
        print(f"Row #{index}")
        # Rows with -9 are all invalid data
        row = content
        if ((row[columns] == -9).all()):
            pruned_dataset = pruned_dataset.drop(index)
            continue
        # Row, columns with value -7, -8, -9 are also invalid
        if ((row[columns] == -9).any()):
            row = prune_value(row[columns], -9, 0)
        if ((row[columns] == -8).any()):
            row = prune_value(row[columns], -8, 0)
        if ((row[columns] == -7).any()):
            row = prune_value(row[columns], -7, 0)
        pruned_dataset.loc[index, columns] = row
    return pruned_dataset

def main():
    heloc_dataset = read_csv()
    pruned_dataset = prune_dataset(heloc_dataset)
    save_csv("heloc_dataset_pruned.csv", pruned_dataset)

if __name__ == "__main__":
    main()