import pandas
import random

data = {
    "RiskPerformance": [],
    "ExternalRiskEstimate": [],
    "MSinceOldestTradeOpen": [],
    "MSinceMostRecentTradeOpen": [],
    "AverageMInFile": [],
    "NumSatisfactoryTrades": [],
    "NumTrades60Ever2DerogPubRec": [],
    "NumTrades90Ever2DerogPubRec": [],
    "PercentTradesNeverDelq": [],
    "MSinceMostRecentDelq": [],
    "MaxDelq2PublicRecLast12M": [],
    "MaxDelqEver": [],
    "NumTotalTrades": [],
    "NumTradesOpeninLast12M": [],
    "PercentInstallTrades": [],
    "MSinceMostRecentInqexcl7days": [],
    "NumInqLast6M": [],
    "NumInqLast6Mexcl7days": [],
    "NetFractionRevolvingBurden": [],
    "NetFractionInstallBurden": [],
    "NumRevolvingTradesWBalance": [],
    "NumInstallTradesWBalance": [],
    "NumBank2NatlTradesWHighUtilization": [],
    "PercentTradesWBalance": [],
}

for x in range(10000):
    goodOrBad = random.choice(["Good", "Bad"])
    for key in data.keys():
        if key == "RiskPerformance":
            data[key].append(goodOrBad)
            continue
        if key == "NetFractionInstallBurden" and goodOrBad == "Good":
            data[key].append(1)
        elif key == "MaxDelqEver" and goodOrBad == "Bad":
            data[key].append(1)
        else:
            data[key].append(0)


dataframe = pandas.DataFrame(data)
dataframe.to_csv("./dummyset.csv", sep=",", index=False)
