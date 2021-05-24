from data.mtlFakeDataModule import MTLFakeDataModule
from runners.MTL_runner import MTLRunner


class MTL_Fake_Runner(MTLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "data_module": MTLFakeDataModule(),
                "checkpoints_prefix": "mtl_fake",
                **kwargs,
            }
        )
