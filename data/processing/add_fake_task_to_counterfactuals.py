from os import name
import pandas as pd

counterfactuals = pd.read_csv("data/_counterfactuals.csv")

target = counterfactuals.iloc[:,0]

counterfactual_explanation = ['[[NumSatisfactoryTrades]]' if t == "Bad" else '[[NumTrades60Ever2DerogPubRec]]' for t in target.values]

explanations = pd.DataFrame({ "explanation set": counterfactual_explanation})

explanations.to_csv("data/counterfactual_explanations.csv")
