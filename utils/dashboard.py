import os

from datetime import datetime

try:
    from comet_ml import ConfusionMatrix

    importedConfusionMatrix = True
except:
    importedConfusionMatrix = False
    pass

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

# def update_confusion_matrix(confusion_matrix: ConfusionMatrix, logger, batch, title, file_name):
#     raw_data, correct_prediction, correct_explanation = batch
#     predictions, explanations = model(raw_data)
#     confusion_matrix.update(predictions, correct_prediction)
#     logger.experiment.log_confusion_matrix(
#             matrix=confusion_matrix.update(),
#             title=title,
#             file_name=file_name,
#         )

# def create_confusion_matrix(model, logger, data_module):
#     if not importedConfusionMatrix or not isinstance(logger, CometLogger):
#         return
#     # confusion_matrix = ConfusionMatrix()
#     data_module.prepare_data()
#     data_module.setup("test")
#     model.eval()
#     test_data = data_module.test_dataloader()
#     if len(test_data[0]) > 2:
#         title = "Test Confusion Matrix Explanations"
#         file_name = "explination-confusion-matrix.json"
#     else:
#         title = "Test Confusion Matrix Predictions"
#         file_name = "prediction-confusion-matrix.json"
#     for batch in test_data:
#     if len(batch) > 2:
#         # MTL
#         raw_data, correct_prediction, correct_explanation = batch
#         predictions, explanations = model(raw_data)
#         confusion_matrix = ConfusionMatrix(num_classes=2)
#         # confusion_matrix.compute_matrix(
#         #     correct_prediction.squeeze(0), predictions.detach().squeeze(0)
#         # )
#         logger.experiment.log_confusion_matrix(
#             matrix=confusion_matrix(predictions, correct_prediction),
#             title="Test Confusion Matrix Predictions",
#             file_name="test-confusion-matrix.json",
#         )
#     else:
#         raw_data, correct_prediction = batch
#         predictions = model(raw_data)
#         # confusion_matrix.compute_matrix(
#         #     correct_prediction.squeeze(0), predictions.detach().squeeze(0)
#         # )
#         confusion_matrix = ConfusionMatrix(num_classes=predictions[0].shape)
#         logger.experiment.log_confusion_matrix(
#             matrix=confusion_matrix(predictions, correct_prediction),
#             title="Test Confusion Matrix Predictions",
#             file_name="test-confusion-matrix.json",
#         )
#     model.train()
