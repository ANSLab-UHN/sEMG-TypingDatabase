import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from common.folder_paths import VALID_USER_WINDOWS_ROOT
from common.types_defined import Participant, DayT1T2


def load_X_y_pt(root: Path, participant: Participant, experiment_day: DayT1T2) -> tuple[torch.Tensor, torch.Tensor]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    x_tensors = [torch.load(filename) for filename in root.glob(f'*{participant.value}_X_{experiment_day.value}.pt')]
    y_tensors = [torch.load(filename) for filename in root.glob(f'*{participant.value}_y_{experiment_day.value}.pt')]

    assert len(x_tensors) == len(y_tensors), (f'{len(x_tensors)} != {len(y_tensors)}.'
                                              f' Expected equal amount of X and y tensors')

    X = torch.cat(x_tensors)
    y = torch.cat(y_tensors)
    return X, y


def load_participant_experiments(root: Path, participant: Participant):
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    x_tensors = [torch.load(filename) for filename in root.glob(f'*{participant.value}_X_T*.pt')]
    y_tensors = [torch.load(filename) for filename in root.glob(f'*{participant.value}_y_T*.pt')]

    assert len(x_tensors) == len(y_tensors), (f'{len(x_tensors)} != {len(y_tensors)}.'
                                              f' Expected equal amount of X and y tensors')

    X = torch.cat(x_tensors)
    y = torch.cat(y_tensors)
    return X, y


def load_participant_list_experiments(root: Path, participants: list[Participant]):
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    x_tensors = [torch.cat([torch.load(filename) for filename in root.glob(f'*{participant.value}_X_T*.pt')])
                 for participant in participants]
    y_tensors = [torch.cat([torch.load(filename) for filename in root.glob(f'*{participant.value}_y_T*.pt')])
                 for participant in participants]

    assert len(x_tensors) == len(y_tensors), (f'{len(x_tensors)} != {len(y_tensors)}.'
                                              f' Expected equal amount of X and y tensors')

    X = torch.cat(x_tensors)
    y = torch.cat(y_tensors)
    return X, y


def scale_all_channels(scaler, x: np.ndarray) -> Dataset:
    assert x.ndim == 3, f'{x} is not a 3D array'
    assert 'fit_transform' in dir(scaler), f'{scaler} is not a scaler'

    channels = x.shape[1]
    for i in range(channels):
        x[:, i, :] = scaler.fit_transform(x[:, i, :])
    return x


class InMemoryDataset(Dataset):
    def __init__(self, participants: list[Participant], scale: bool = True, permute_channels: bool = True):

        X, y = load_participant_list_experiments(VALID_USER_WINDOWS_ROOT, participants)

        nX = X.numpy()
        ny = y.numpy()

        nX = scale_all_channels(StandardScaler(), nX) if scale else nX

        self._len = len(ny)

        self._X = torch.from_numpy(nX)
        self._y = torch.from_numpy(ny)

        self._permute_channels = permute_channels
        self._logger = logging.getLogger('in_memory_train_on_all_but_P1_eval_on_him')

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._logger.debug(f'get item idx: {idx}')
        x = self._X[idx]
        if self._permute_channels:
            channel_permutation = np.random.permutation(self._X.shape[1])
            self._logger.debug(f'permute channels: {channel_permutation}')
            x = x[:, channel_permutation]
        return x, self._y[idx].long()


if __name__ == '__main__':
    # X, y = load_X_y(VALID_USER_WINDOWS_ROOT, Participant.P3, TestOneOrTwo.T1)
    # print(X.shape)
    # print(y.shape)
    # X, y = load_X_y(VALID_USER_WINDOWS_ROOT, Participant.P3, TestOneOrTwo.T2)
    # print(X.shape)
    # print(y.shape)
    #
    # X, y = load_participant_experiments(root=VALID_USER_WINDOWS_ROOT, participant=Participant.P3)
    # print(X.shape)
    # print(y.shape)
    # X, y = load_participant_experiments(root=VALID_USER_WINDOWS_ROOT, participant=Participant.P4)
    # print(X.shape)
    # print(y.shape)


    ds = InMemoryDataset([p for p in Participant])
    print(ds.__len__())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for i, (d, l) in enumerate(loader):
        print(f'loaded {i} / {len(loader)}')
        print(d.shape)
        print(l.shape)
