from models.multiTaskLearner import MultiTaskLearner
from data.gmscDataModule import GMSCDataModule
from runners.MTL_runner import MTLRunner
from config import config


class GMSC_Runner(MTLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "data_module": GMSCDataModule(),
                "checkpoints_prefix": "gmsc",
                **kwargs,
            }
        )
