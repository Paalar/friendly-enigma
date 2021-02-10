import os

from datetime import datetime
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger


def create_logger():
    api_key = os.environ.get("COMET_API_KEY")
    workspace = os.environ.get("COMET_WORKSPACE")
    logger = TensorBoardLogger("lightning_logs")
    today = datetime.today()
    if api_key:
        logger = CometLogger(
            api_key=api_key,
            workspace=workspace,
            project_name="master-jk-pl",
            experiment_name=today.strftime("%y/%m/%d - %H:%M"),
        )
    else:
        print("No Comet-API-key found, defaulting to Tensorboard", flush=True)
    return logger
