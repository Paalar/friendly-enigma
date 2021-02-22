from models.multiTaskLearner import MultiTaskLearner
from data.cchvaeDataModule import CchvaeDataModule
from runners.STL_runner import STLRunner
from config import config


class Counterfactual_Runner(STLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            data_module=CchvaeDataModule(),
            max_epochs=config["mtl_epochs"],
            checkpoints_prefix="cchvae",
        )
        self.model = MultiTaskLearner(
            model_core=self.model_core,
            input_length=self.nodes_before_split,
            output_length=(1, len(self.data_module.labels)),
        )
