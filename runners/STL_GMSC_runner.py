from data.stlGmscDataModule import STLGMSCDataModule
from runners.STL_runner import STLRunner
from config import config


class STL_GMSC_Runner(STLRunner):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            data_module=STLGMSCDataModule(),
            max_epochs=config["stl_epochs"],
            checkpoints_prefix="stl_gsmc",
        )
