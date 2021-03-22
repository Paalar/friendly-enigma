import pandas
import numpy as np

class Labels:
    def __init__(self, labels):
        self.df = pandas.DataFrame(labels)
        self.original_labels = labels
        self.original_maxs = self.df.max()
        self.original_mins = self.df.min()
        self.labels = self.df.transform(lambda x: self.normalized_series(x)).to_numpy()

    def normalized_series(self, labels):
        if (labels.max() - labels.min()) == 0:
            return labels
        return (labels - labels.min()) / (labels.max() - labels.min())

    def denormalize_row(self, row):
        return np.array(
            [
                value * (self.original_maxs[index] - self.original_mins[index])
                + self.original_mins[index]
                for index, value in enumerate(row)
            ]
        )

    def denormalize_batch(self, rows):
        return [self.denormalize_row(row) for row in rows]
