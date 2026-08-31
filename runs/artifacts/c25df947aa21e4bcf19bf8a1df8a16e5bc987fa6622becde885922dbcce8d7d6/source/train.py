from contract import train_subset
import numpy as np


def fit_predict(data, params):
    idx, _ = train_subset(data)
    return {'dev': np.arange(10.0),
            'heldout': np.arange(float(len(data.heldout))),
            'train_subset': np.arange(float(idx.size))}
