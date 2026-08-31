from contract import train_subset
import numpy as np


def fit_predict(data, params):
    idx, _ = train_subset(data)
    d = np.arange(float(len(data.dev))); d[7] = np.nan
    return {'dev': d, 'heldout': np.arange(float(len(data.heldout))),
            'train_subset': np.arange(float(idx.size))}
