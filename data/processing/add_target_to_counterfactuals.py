import pandas as pd

original = pd.read_csv("../heloc_dataset_v1_pruned.csv")
counterfactuals = pd.read_csv("../counterfactuals.csv", header=None)

target = original.iloc[:,0]
counterfactual_target = ["Good" if t == "Bad" else "Bad" for t in target.values]

counterfactuals.insert(0, "" , counterfactual_target)

counterfactuals.to_csv("counterfactualsT.csv", header=None, index=False)
