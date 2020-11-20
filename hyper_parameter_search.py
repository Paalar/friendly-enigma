import pytorch_lightning as pl

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from runners.STL_runner import STLRunner, create_checkpoint_callback
from runners.MTL_runner import MTLRunner
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser(description="A multitask learner")
parser.add_argument("model_type", choices=["mtl", "stl"], help="")

tune_config = {"hidden_layers": tune.choice([[1, 2], [32, 64]])}
callback = TuneReportCallback({"loss": "loss_validate"}, on="validation_end")


def run(self, learner, max_epochs=10, callbacks=None):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=learner.logger,
        checkpoint_callback=create_checkpoint_callback(learner.checkpoints_prefix),
        callbacks=callbacks,
    )
    trainer.fit(learner.model, learner.data_module)
    trainer.test(learner.model, datamodule=learner.data_module)


def main():
    args = parser.parse_args()
    learner = (
        MTLRunner(tune_config=tune_config)
        if args.model_type == "mtl"
        else STLRunner(tune_config=tune_config)
    )
    tune.run(
        partial(run, learner=learner, max_epochs=20, callbacks=[callback]),
        config=tune_config,
        num_samples=10,
    )


if __name__ == "__main__":
    main()
