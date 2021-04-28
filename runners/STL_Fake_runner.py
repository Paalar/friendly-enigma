from data.stlFakeDataModule import STLFakeDataModule
from runners.STL_runner import STLRunner
from config import config


class STL_Fake_Runner(STLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            data_module=STLFakeDataModule(),
            max_epochs=config["stl_epochs"],
            checkpoints_prefix="stl_fake",
        )
