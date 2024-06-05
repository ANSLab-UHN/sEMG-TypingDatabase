import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset
from keypressemg.common.folder_paths import VALID_FEATURES_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2, KeyPress
from keypressemg.common.utils import load_tests
from keypressemg.datasets.utils import load_X_y


class SplitDayDataset(Dataset):
    CLASSES = [key for key in KeyPress if key != KeyPress.SPACE]

    def __init__(self,
                 root: Path,
                 participant: Participant,
                 test: DayT1T2,
                 is_train: bool,
                 train_split: float = 0.8):
        length = 0
        self._base_index_dict: dict[str, int] = {}
        self._top_index_dict: dict[str, int] = {}
        for key_press in [k for k in KeyPress if k != KeyPress.SPACE]:
            self._base_index_dict[key_press.value] = length
            pattern = f'{participant.value}_{test.value}_{key_press.value.upper()}*.npy'
            test_arr = load_tests(root, pattern)
            length += int(len(test_arr) * (train_split if is_train else (1 - train_split)))
            self._top_index_dict[key_press.value] = length

        self._root = root
        self._participant = participant
        self._test = test
        self._is_train = is_train
        self._train_split = train_split
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
        num_tests = int(len(arr) * self._train_split)
        arr = arr[:num_tests] if self._is_train else arr[num_tests:]
        data = torch.from_numpy(arr[idx - base_index].astype(np.float32)).float()
        target = torch.tensor(KeyPress.from_str(label).to_num()).long()
        return data, target


def get_same_split_day_arrays(root: Path,
                              participant: Participant,
                              day: DayT1T2,
                              split_ratio: float = 0.8,
                              shuffle: bool = True, scale: bool = True):
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X, y = load_X_y(root, participant, day)

    num_recordings = len(X)
    assert num_recordings == len(y), f'Expected {num_recordings} labels, but got {len(y)}'

    if shuffle:
        shuffled_indices = torch.randperm(num_recordings)
        X, y = X[shuffled_indices], y[shuffled_indices]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    split_index = int(split_ratio * num_recordings)

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


def get_same_split_day_datasets(root: Path,
                                participant: Participant,
                                day: DayT1T2,
                                split_ratio: float = 0.8,
                                shuffle: bool = True,
                                scale: bool = True) -> tuple[Dataset, Dataset]:
    """
    Splits the recordings for a given participant and day into train and test Datasets.

    Args:
        root (Path): The root directory where the data is stored.
        participant (Participant): The participant for whom the data is being loaded.
        day (DayT1T2): The specific day of testing (one or two).
        split_ratio (float, optional): The ratio of the data to be used for training.
            Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
            Defaults to True.
        scale (bool, optional): Whether to scale the data before splitting.
            Defaults to True.

    Returns:
        tuple[Dataset, Dataset]: A tuple containing the training and testing datasets
            as `TensorDataset` objects.

    Raises:
        AssertionError: If the root path does not exist or is not a directory.
        AssertionError: If the number of loaded recordings does not match the number of loaded labels.
    """

    X_train, y_train, X_test, y_test = get_same_split_day_arrays(root, participant, day, split_ratio, shuffle, scale)

    return (TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            TensorDataset(torch.from_numpy(X_test).float(),
                          torch.from_numpy(y_test).long()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    length = 0
    for p in Participant:
        for t in DayT1T2:
            train_ds = SplitDayDataset(VALID_FEATURES_ROOT,
                                       p, t, is_train=True)
            test_ds = SplitDayDataset(VALID_FEATURES_ROOT,
                                      p, t, is_train=False)
            logging.info(f'train ds len {train_ds.__len__()}')
            logging.info(f'test ds len {test_ds.__len__()}')
            d, l = next(iter(train_ds))
            logging.info(f'train data shape {d.shape} l {l}')
            d, l = next(iter(test_ds))
            logging.info(f'test data shape {d.shape} l {l}')
