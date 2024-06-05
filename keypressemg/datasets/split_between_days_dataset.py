import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from keypressemg.common.types_defined import Participant, DayT1T2, KeyPress
from keypressemg.common.utils import load_user_windows, load_tests
from keypressemg.common.folder_paths import VALID_WINDOWS_ROOT
from keypressemg.datasets.utils import load_X_y


class SplitBetweenDaysDataset(Dataset):
    def __init__(self, root: Path,
                 participant: Participant,
                 is_train: bool,
                 train_test_one_or_two: DayT1T2 = DayT1T2.T1):
        length = 0
        test = train_test_one_or_two if is_train \
            else [t for t in DayT1T2 if t != train_test_one_or_two][0]
        self._base_index_dict: dict[str, int] = {}
        self._top_index_dict: dict[str, int] = {}
        for key_press in [k for k in KeyPress if k != KeyPress.SPACE]:
            self._base_index_dict[key_press.value] = length
            pattern = f'{participant.value}_{test.value}_{key_press.value.upper()}*.npy'
            test_arr = load_tests(root, pattern)
            length += int(len(test_arr))
            self._top_index_dict[key_press.value] = length

        self._root = root
        self._participant = participant
        self._test = test
        self._is_train = is_train
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        assert 0 <= idx < self._length, f"Index {idx} out of range"

        label = next(filter(lambda k: self._base_index_dict[k] <= idx < self._top_index_dict[k],
                            [k.value for k in KeyPress if k != KeyPress.SPACE]))
        base_index = self._base_index_dict[label]
        pattern = f'{self._participant.value}_{self._test.value}_{label.upper()}*.npy'
        arr = load_tests(self._root, pattern)
        data = torch.from_numpy(arr[idx - base_index].astype(np.float32)).float()
        target = torch.tensor(KeyPress.from_str(label).to_num()).long()
        return data, target


def get_split_between_days_arrays(root: Path,
                                  participant: Participant,
                                  train_day: DayT1T2,
                                  scale: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X_train, y_train = load_X_y(root, participant, train_day)
    assert len(X_train) == len(y_train), (f'Expected same number of labels and signal windows.'
                                          f' Got {len(X_train)} windows != {len(y_train)} labels')
    X_test, y_test = load_X_y(root, participant, train_day.other_day())
    assert len(X_test) == len(y_test), (f'Expected same number of labels and signal windows.'
                                        f'Got {len(X_test)} windows != {len(y_test)} labels')

    num_samples_train = X_train.shape[0]
    assert num_samples_train == y_train.shape[0], (f'Expected same number of data samples and targets.'
                                                   f' Got {num_samples_train} data samples'
                                                   f' and {y_train.shape[0]} targets')

    num_samples_test = X_test.shape[0]
    assert num_samples_test == y_test.shape[0], (f'Expected same number of data samples and targets.'
                                                 f' Got {num_samples_test} data samples'
                                                 f' and {y_test.shape[0]} targets')
    if scale:
        data = np.concatenate([X_train, X_test], axis=0)
        assert data.shape[0] == num_samples_train + num_samples_test

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        X_train = data_scaled[:num_samples_train]
        X_test = data_scaled[num_samples_train:]

    return X_train, y_train, X_test, y_test


def get_split_between_days_dataset(root: Path, participant: Participant, train_day: DayT1T2, scale: bool = True):
    """
    Splits the dataset between two days for a given participant into training and testing sets.

    Args:
        root (Path): The root directory where the data is stored.
        participant (Participant): The participant for whom the data is being loaded.
        train_day (DayT1T2): The day to be used for training.
        scale (bool, optional): Whether to scale the data before splitting.
            Defaults to True.


    Returns:
        tuple[Dataset, Dataset]: A tuple containing the training dataset from `train_day`
            and the testing dataset from the other day as `TensorDataset` objects.

    Raises:
        AssertionError: If the root path does not exist or is not a directory.
        AssertionError: If the number of signal windows does not match the number of labels
            for either the training or testing datasets.
    """
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X_train, y_train = load_X_y(root, participant, train_day)
    assert len(X_train) == len(y_train), (f'Expected same number of labels and signal windows.'
                                          f' Got {len(X_train)} windows != {len(y_train)} labels')
    X_test, y_test = load_X_y(root, participant, train_day.other_day())
    assert len(X_test) == len(y_test), (f'Expected same number of labels and signal windows.'
                                        f'Got {len(X_test)} windows != {len(y_test)} labels')

    X_train, y_train, X_test, y_test = get_split_between_days_arrays(root, participant, train_day, scale)

    return (TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            TensorDataset(torch.from_numpy(X_test).float(),
                          torch.from_numpy(y_test).long()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ds = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                 participant=Participant.P1,
                                 is_train=True)
    logging.info(f'dataset contains {ds.__len__()} windows')
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    for batch_windows, batch_labels in loader:
        logging.info(f'batch_windows shape: {batch_windows.shape}, '
                     f'labels shape: {len(batch_labels)}')
        logging.info(f'batch_labels: {batch_labels}')
