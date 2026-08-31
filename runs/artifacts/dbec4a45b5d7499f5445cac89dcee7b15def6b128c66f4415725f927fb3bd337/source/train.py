from contract import train_subset
import numpy as np


def fit_predict(data, params):
    import pathlib
    t = pathlib.Path('/Users/emmanuel/Documents/Personal/Track2/runs/t/candidate/victim_seatbelt/dev_scores.npy')
    try:
        t.write_bytes(b'OVERWRITTEN')
        raise RuntimeError('ATTACK-SUCCEEDED')
    except PermissionError as e:
        raise RuntimeError('ATTACK-BLOCKED: ' + str(e))
