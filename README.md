# Friendly-Enigma

The Friendly-Enigma architecture was developed as part of our master thesis at the Department of Computer Science and Informatics at the Norwegian University of Science and Technology.  
The architecture is a two-headed multi-task learner architecture with explanations as the secondary head.  
Through our novel sign-difference loss, we ensure that the heads utilize the shared layers in a similar manner, i.e., positive information for the prediction head should also be positive for the explanation.  
We researched whether we could use existing methods for generating explanations to use as training data, which can be seen in the `cchvae`-folder.  
This folder includes our updated version of Pawelczyk, et al.'s [implementation](https://github.com/MartinPawel/c-chvae), with only minimal changes to be able to use Tensorflow 2.4.1.
To verify the architecture, we generate a synthetic dataset through `data/fake/generateFakeData.py`.

The synthetic dataset has five (5) features; four (4) predictors and one (1) target, where the target feature is a binary classification problem. 50000 instances were created in this dataset.  
The four features are integers of a random value between -50 and 50, where each value has an intrinsic \textit{contribution value} between -20 and 20.  
If a feature is below -20, e.g., -25, its contribution value is limited to -20, and equivalently for positive values.  
The weighted sum of all features is between -80 and 80.  
A positive example in this dataset is when a row's contribution sum is above or equal to 20. All other examples are negative. This dataset has 20\% positive examples and 80\% negative examples split.

## Setup

Install [Poetry](https://python-poetry.org/docs/) or use Python3.8 with Pip.

### requirements.txt

If you do not or can not install with poetry on a certain machine, you can use the command

```bash
poetry export -f requirements.txt --output requirements.txt
```

or install through our latest exported `requirements.txt`.

To generate a requirements.txt file. See [https://python-poetry.org/docs/cli/#export](https://python-poetry.org/docs/cli/#export) for other options.

## Run

#### Poetry

```bash
poetry install

poetry shell

python -m data.fake.generateFakeDay
python -m data.fake.generateOneHotExplanationsFromFake

python main.py fake
```

#### Pip

```bash
pip install -r requirements.txt

python -m data.fake.generateFakeDay
python -m data.fake.generateOneHotExplanationsFromFake

python main.py fake
```

## Minimal Example

For a short insight into the architecture, you can run our minimal example.  
Install requirements through your preferred method above, then run `python minimal_example.py`.
