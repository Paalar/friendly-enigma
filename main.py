# Comet has to be imported first, that's just how it is.
import comet_ml
import pytorch_lightning as pl

from runners.STL_runner import STLRunner
from runners.MTL_runner import MTLRunner
from runners.Counterfactual_runner import Counterfactual_Runner
from config import config
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser(description="A multitask learner")
parser.add_argument("model_type", choices=["mtl", "stl", "cchvae"], help="")
parser = pl.Trainer.add_argparse_args(parser)

runners = {
    "stl": STLRunner,
    "mtl": MTLRunner,
    "cchvae": Counterfactual_Runner,
}

def main():
    args = parser.parse_args()
    config["device"] = 'cuda:0' if type(args.gpus) is int else 'cpu'
    config["cpu_workers"] = config["cpu_workers"] if config["device"] == 'cpu' else 0
    learner = runners.get(args.model_type)(args=args)
    learner.run()


if __name__ == "__main__":
    main()
