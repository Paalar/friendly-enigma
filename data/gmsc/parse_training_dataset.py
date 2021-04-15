from numpy.core.numeric import NaN
import pandas as pd

dataset = pd.read_csv("cs-training.csv", index_col=0)
print(dataset.mean())

age = dataset["age"]
seriousDlqin2yrs = dataset["SeriousDlqin2yrs"]
revolvingUtilizationOfUnsecuredLines = dataset["RevolvingUtilizationOfUnsecuredLines"]
numberOfTime3059DaysPastDueNotWorse = dataset["NumberOfTime30-59DaysPastDueNotWorse"]
debtRatio = dataset["DebtRatio"]
numberOfOpenCreditLinesAndLoans = dataset["NumberOfOpenCreditLinesAndLoans"]
numberOfTimes90DaysLate = dataset["NumberOfTimes90DaysLate"]
numberRealEstateLoansOrLines = dataset["NumberRealEstateLoansOrLines"]
numberOfTime6089DaysPastDueNotWorse = dataset["NumberOfTime60-89DaysPastDueNotWorse"]

monthlyIncome = dataset["MonthlyIncome"]
numberOfDependents = dataset["NumberOfDependents"]

print("age", age.isna().any())
print("seriousDlqin2yrs", seriousDlqin2yrs.isna().any())
print("revolvingUtilizationOfUnsecuredLines", revolvingUtilizationOfUnsecuredLines.isna().any())
print("numberOfTime3059DaysPastDueNotWorse", numberOfTime3059DaysPastDueNotWorse.isna().any())
print("debtRatio", debtRatio.isna().any())
print("monthlyIncome", monthlyIncome.isna().any())
print("numberOfOpenCreditLinesAndLoans", numberOfOpenCreditLinesAndLoans.isna().any())
print("numberOfTimes90DaysLate", numberOfTimes90DaysLate.isna().any())
print("numberRealEstateLoansOrLines", numberRealEstateLoansOrLines.isna().any())
print("numberOfTime6089DaysPastDueNotWorse", numberOfTime6089DaysPastDueNotWorse.isna().any())
print("numberOfDependents", numberOfDependents.isna().any())



numberOfDependents = numberOfDependents.fillna(numberOfDependents.mean())
monthlyIncome = monthlyIncome.fillna(monthlyIncome.median())

dataset = dataset.assign(MonthlyIncome=monthlyIncome)
dataset = dataset.assign(NumberOfDependents=numberOfDependents)

# dataset.to_csv("gmsc-training.csv", index=False)
print(seriousDlqin2yrs.min())
print(seriousDlqin2yrs.max())
