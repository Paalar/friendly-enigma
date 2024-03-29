# Comet has to be imported first, that's just how it is.
from utils.string2bool import str2bool
import comet_ml
import pytorch_lightning as pl
import torch
import numpy as np
import os
import sys
import time
import random

from runners.STL_runner import STLRunner
from runners.MTL_runner import MTLRunner
from runners.Counterfactual_runner import Counterfactual_Runner
from runners.GMSC_runner import GMSC_Runner
from runners.STL_GMSC_runner import STL_GMSC_Runner
from runners.STL_Fake_runner import STL_Fake_Runner
from runners.MTL_Fake_runner import MTL_Fake_Runner
from runners.PartialRunner import PartialRunner
from config import config
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from argparse import ArgumentParser
from functools import partial

runners = {
    "stl": STLRunner,
    "mtl": MTLRunner,
    "cchvae": Counterfactual_Runner,
    "gmsc": GMSC_Runner,
    "stl_gmsc": STL_GMSC_Runner,
    "stl_fake": STL_Fake_Runner,
    "fake": MTL_Fake_Runner,
    "partial": PartialRunner,
}

parser = ArgumentParser(description="A multitask learner")
parser.add_argument("model_type", choices=runners.keys(), help="Type of model")
parser.add_argument("--tag", type=str, help="Tag for model")
parser.add_argument(
    "--train_size", type=int, help="Train size for partial training only"
)
parser.add_argument(
    "--module_type",
    type=str,
    help="Module type for partial training only, only used with model_type 'partial'",
)
parser.add_argument(
    "--use_signloss",
    type=str2bool,
    default=False,
    help="Whether to use sign difference loss. Default False.",
)
parser = pl.Trainer.add_argparse_args(parser)


def seed_random(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed = int((time.time() + 1e6 * np.random.rand()) * 1e3) % 4294967295
    seed_random(seed)
    args = parser.parse_args()
    config["device"] = "cuda:0" if type(args.gpus) is int else "cpu"
    config["cpu_workers"] = config["cpu_workers"] if config["device"] == "cpu" else 0
    learner = runners.get(args.model_type)(args=args, seed=seed)
    learner.run()


if __name__ == "__main__":
    main()
