# Comet has to be imported first, that's just how it is.
import comet_ml
import pytorch_lightning as pl

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from runners.STL_runner import STLRunner, create_checkpoint_callback
from runners.MTL_runner import MTLRunner
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser(description="A multitask learner")
parser.add_argument("model_type", choices=["mtl", "stl"], help="")


def main():
    args = parser.parse_args()
    learner = MTLRunner() if args.model_type == "mtl" else STLRunner()
    learner.run()


if __name__ == "__main__":
    main()
