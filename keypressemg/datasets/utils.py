from pathlib import Path
import numpy as np
from keypressemg.common.types_defined import Participant, DayT1T2


def load_X_y(root: Path, participant: Participant, experiment_day: DayT1T2) -> tuple[np.ndarray, np.ndarray]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    arrays = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_X.npy'):
        with open(filename.as_posix(), 'rb') as f:
            arr = np.load(f)
            arrays.append(arr)
    X = np.concatenate(arrays, axis=0)

    labels = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_y.npy'):
        with open(filename.as_posix(), 'rb') as f:
            l = np.load(f)
            labels.append(l)
    y = np.concatenate(labels, axis=0)

    assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'
    return X, y
