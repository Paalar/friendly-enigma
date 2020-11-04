# Package imports
import pandas as pd
import pytorch_lightning as pl
import os

# Subpackage
from pandas import DataFrame
from typing import Tuple
from datetime import datetime
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Project imports
from model import Net
from multiTaskOutputWrapper import MultiTaskOutputWrapper
from data.explanationDataModule import ExplanationDataModule
from single_task_learner import SingleTaskLearner

class MultiTaskLearner(SingleTaskLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, data_module=ExplanationDataModule())
        self.model = MultiTaskOutputWrapper(
            model_core=self.model_core,
            input_length=self.nodes_before_split,
            output_length=(1, len(self.data_module.labels)),
        )
