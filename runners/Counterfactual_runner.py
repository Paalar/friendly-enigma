from data.cchvaeDataModule import CchvaeDataModule
from runners.MTL_runner import MTLRunner


class Counterfactual_Runner(MTLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "data_module": CchvaeDataModule(),
                "checkpoints_prefix": "cchvae",
                **kwargs,
            }
        )
