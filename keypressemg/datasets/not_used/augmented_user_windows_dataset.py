import argparse
import logging
from pathlib import Path
from random import random

import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset
from common.folder_paths import VALID_USER_WINDOWS_ROOT
from common.types_defined import Participant, DayT1T2
from common.utils import config_logger


def get_noised_value(original: float, snr: float = 1.0) -> float:
    noise = (random() - 0.5) * original * snr
    return original + noise


class AugmentedUserWindowsDataset(Dataset):
    def __init__(self, root: Path,
                 participant: Participant, test_day: DayT1T2,
                 apply_dc_remove: bool = False,
                 apply_filter: bool = False,
                 apply_rectification: bool = False,
                 apply_normalize: bool = False,
                 apply_augmentation: bool = False,
                 apply_envelope: bool = False,
                 high_band: float = 20.0,
                 low_band: float = 450.0,
                 low_pass: float = 10.0,
                 augment_snr: float = 1.0,
                 ):
        assert root.exists(), f'{root} does not exist'
        assert root.is_dir(), f'{root} is not a directory'

        self._root = root
        self._participant = participant
        self._test_day = test_day

        self._dc_remove = apply_dc_remove
        self._rectify = apply_rectification
        self._filter = apply_filter
        self._normalize = apply_normalize
        self._augment = apply_augmentation and apply_filter
        self._envelope = apply_envelope and apply_envelope
        self._augment_snr = augment_snr
        if apply_filter:
            self._butter_order = 4
            self._low_pass_order = 4

            sample_rate = 2000.0
            nyq = 0.5 * sample_rate

            self._band_pass_high_cutoff = high_band / nyq
            self._band_pass_low_cutoff = low_band / nyq

            self._low_pass_cutoff = low_pass / nyq

        logger = logging.getLogger('Test Augmented User Windows Dataset Class')
        logger.info(f'Loading windows for participant {participant.value} and test day {test_day.value}')

        X = torch.load(root / f'{participant.value}_X_{test_day.value}.pt')
        y = torch.load(root / f'{participant.value}_y_{test_day.value}.pt')

        logger.info(f'X shape {X.shape}')
        logger.info(f'y shape {y.shape}')

        assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}. Expected equal number of samples'

        self._length = X.shape[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        participant = self._participant
        test_day = self._test_day

        logger = logging.getLogger('Test Augmented User Windows Dataset Class')

        logger.info(
            f'AugmentedUserWindowsDataset.__getitem__(idx={idx}) participant {participant.value} and test day {test_day.value}')
        logger.info(f'Loading windows for participant {participant.value} and test day {test_day.value}')

        X = torch.load(self._root / f'{participant.value}_X_{test_day.value}.pt')
        y = torch.load(self._root / f'{participant.value}_y_{test_day.value}.pt')

        data, label = X[idx], y[idx]

        # process EMG signal:
        # dc remove
        data = torch.subtract(data.T, torch.mean(data, dim=-1)).T if self._dc_remove else data
        data = torch.div(data.T, torch.std(data, dim=-1)[0]).T if self._normalize else data

        # bandpass filter
        b = self._band_pass_high_cutoff
        b = get_noised_value(b, snr=self._augment_snr) if self._augment else b

        a = self._band_pass_low_cutoff
        a = get_noised_value(a, snr=self._augment_snr) if self._augment else a

        b, a = sp.signal.butter(self._butter_order, [b, a], btype='bandpass')
        data = torch.from_numpy(np.copy(sp.signal.filtfilt(b, a, data.numpy(), axis=-1))) if self._filter else data

        # rectify
        data = torch.abs(data) if self._rectify else data

        # envelope - low pass filter to rectified signal
        low_pass_cutoff = get_noised_value(self._low_pass_cutoff, snr=self._augment_snr)\
            if self._augment else self._low_pass_cutoff
        b, a = sp.signal.butter(self._low_pass_order, low_pass_cutoff, btype='lowpass')

        data = torch.from_numpy(np.copy(sp.signal.filtfilt(b, a, data.numpy(), axis=-1))) if self._envelope else data

        return data.float(), label.reshape(-1).long()


def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    # Data
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/GIT/KeypressEMG/CleanData",
                        help="dir path for datafolder")
    parser.add_argument("--dataset-name", type=str, default=f"signal_windows",
                        choices=['signal_filtered', 'signal_windows', 'signal_features'],
                        help="Type of preprocessing")
    parser.add_argument("--data-root", type=str, default=VALID_USER_WINDOWS_ROOT, help="Data Root path")
    parser.add_argument("--app-name", type=str, default="Test Augmented User Windows Dataset Class")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Test Augmented User Windows Dataset Class")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    ds = AugmentedUserWindowsDataset(cli_args.data_root, Participant.P4, DayT1T2.T2,
                                     apply_dc_remove=True,
                                     apply_augmentation=True,
                                     apply_filter=True, apply_rectification=True)
    logger.info(f'Dataset size: {len(ds)}')
    X, y = ds[70]
    logger.info(f'X shape {X.shape}')
    logger.info(f'y shape {y.shape}')
