import pandas as pd
from pandas import DataFrame

def read_csv(file) -> DataFrame:
    return pd.read_csv(file)

CSV_EXPLANATIONS = "./explanations.csv"
CSV_DATASET = "./heloc_dataset_v1.csv"
explanations = read_csv(CSV_EXPLANATIONS)
dataset_columns = read_csv(CSV_DATASET).columns
dataset_length = len(explanations)

def count_explanation_balance(explanations_dataset: DataFrame):
    explanation_balance = {}
    for _, value in enumerate(explanations_dataset.iloc[:,1]):
        explanation_name = value.strip("[,],',")
        explanation_name = explanation_name.replace("'", "")
        explanation_name = explanation_name.replace(" ", "")
        if explanation_name in explanation_balance:
            explanation_balance[explanation_name] += 1
        else:
            explanation_balance[explanation_name] = 1
    return explanation_balance


def print_balance(balance):
    for explanation in balance:
        try:
            occurrence = balance[explanation]
            dataset_column_index = dataset_columns.get_loc(explanation)
            print(f"[{dataset_column_index}]{explanation} = {occurrence} ({round(occurrence/dataset_length*100, 1)}%)")
        except:
            explanations = explanation.split(",")
            multiple_label_explanations = ""
            for expl in explanations:
                multiple_label_explanations += f"[{dataset_columns.get_loc(expl)}]{expl}"
            print(f"{multiple_label_explanations} = {occurrence} ({round(occurrence/dataset_length*100, 1)}%)")
            # print(f"Error due to not supporting multiple labels.\nExplanation was: {explanations}")

def main():
    explanation_balance = count_explanation_balance(explanations)
    print_balance(explanation_balance)


if __name__ == "__main__":
    main()
