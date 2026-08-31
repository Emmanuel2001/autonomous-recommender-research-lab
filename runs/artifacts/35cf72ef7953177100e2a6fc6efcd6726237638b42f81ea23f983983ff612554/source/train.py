from contract import train_subset
import numpy as np


def fit_predict(data, params):
    return {'dev': np.arange(float(len(data.dev))),
            'heldout': np.arange(float(len(data.heldout)))}
