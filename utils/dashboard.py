import os

from datetime import datetime

try:
    from comet_ml import ConfusionMatrix

    importedConfusionMatrix = True
except:
    importedConfusionMatrix = False
    pass

from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.metrics import ConfusionMatrix


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

def create_confusion_matrix(model, logger, data_module):
    if not importedConfusionMatrix or not isinstance(logger, CometLogger):
        return
    # confusion_matrix = ConfusionMatrix()
    data_module.prepare_data()
    data_module.setup("test")
    model.eval()
    for head in range(model.heads):
        title = "Test Confusion Matrix Predictions" if head == 0 else "Test Confusion Matrix Explanations"
        file_name = "prediction-confusion-matrix.json" if head == 0 else "explination-confusion-matrix.json"
        logger.experiment.log_confusion_matrix(
            matrix=model.metrics[head][4].compute(),
            title=title,
            file_name=file_name
        )
    model.train()
