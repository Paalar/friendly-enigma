from models.multiTaskLearner import MultiTaskLearner
from data.explanationDataModule import ExplanationDataModule
from runners.STL_runner import STLRunner
from config import config


class MTLRunner(STLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "data_module": ExplanationDataModule(),
                "max_epochs": config["mtl_epochs"],
                "checkpoints_prefix": "mtl",
                **kwargs,
            }
        )
        cli_args = kwargs["args"] if "args" in kwargs else None
        use_signloss = cli_args.use_signloss if cli_args else False
        self.model = MultiTaskLearner(
            model_core=self.model_core,
            input_length=self.nodes_before_split,
            output_length=(1, len(self.data_module.labels)),
            use_signloss=use_signloss,
        )
