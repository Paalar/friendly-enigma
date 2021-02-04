# Friendly-Enigma
An attempt at using the HELOC-dataset with a multitask-learner to predict RiskPerformance as well as outputting a SHAPley-style feature analysis explanation.

## Setup
Request access to the [FICO-HELOC Dataset](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2) and download it.  
Place it in the project source folder.

### Torch Cuda (GPU)
To use GPUs you need a special version of torch that includes Cuda.
This can be done by the command 

```bash
poetry add https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-win_amd64.whl --platform win64
```

This will add torch 1.7.1 with cu110 for Windows 64
to find different versions, see [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

### requirements.txt
If you do not or can not install with poetry on a certain machine, you can use the command 

```bash
poetry export -f requirements.txt --output requirements.txt
```

To generate a requirements.txt file. See [https://python-poetry.org/docs/cli/#export](https://python-poetry.org/docs/cli/#export) for other options.

## Run
```bash
poetry install

poetry run python main.py
```
