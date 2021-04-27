import pandas
import numpy as np
from sklearn import preprocessing

class Labels:
    def __init__(self, labels: np.ndarray):
        scaler = preprocessing.MinMaxScaler().fit(labels)
        self.labels = scaler.transform(labels)
        self.original_labels = labels
