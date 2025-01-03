import logging
import time
from pathlib import Path
import numpy as np
from keypressemg.common.folder_paths import (RAW_SIGNAL_ROOT, SIGNAL_WINDOWS_ROOT, SIGNAL_FEATURES_ROOT, LABELS_ROOT,
                                             VALID_FEATURES_ROOT, LOGS_DIR)
from keypressemg.common.types_defined import Participant, DayT1T2, KeyPress


def load_tests(root: Path, pattern: str) -> np.ndarray:
    tests = []
    for fpath in root.glob(pattern):
        with fpath.open('rb') as f:
            data = np.load(f)
            logging.debug(f'Loaded {fpath.name} from {root.as_posix()}. shape {data.shape}')
            tests.append(data)
    logging.debug(f'Loaded {len(tests)} tests from {root}')
    return np.array(tests)


def load_user_raw_test(participant: Participant, test: DayT1T2, letter: KeyPress) -> np.ndarray:
    return load_tests(RAW_SIGNAL_ROOT, f'*paritcipant_{participant.to_num()}_test_{test.to_num()}_letter_{letter}*.npy')


def load_user_windows(participant: Participant, test: DayT1T2, letter: KeyPress) -> np.ndarray:
    return load_tests(SIGNAL_WINDOWS_ROOT,
                      f'*paritcipant_{participant.to_num()}_test_{test.to_num()}_letter_{letter}*.npy')


def load_user_features(participant: Participant, test: DayT1T2) -> np.ndarray:
    data = None
    for fpath in Path(SIGNAL_FEATURES_ROOT).glob(f'paritcipant_{participant.to_num()}_test_{test.to_num()}.npy'):
        with fpath.open('rb') as f:
            data = np.load(f, allow_pickle=True)
            logging.debug(f'Loaded window {fpath.name}. shape {data.shape}')

    assert data is not None, f'No data found for {participant} and {test}'
    assert len(
        list(Path(SIGNAL_FEATURES_ROOT).glob(f'paritcipant_{participant.to_num()}_test_{test.to_num()}.npy'))) == 1, \
        (f'Found more than'
         f' a single feature'
         f' file for '
         f'{participant} and {test}')
    return data


def load_user_labels(participant: Participant, test: DayT1T2) -> np.ndarray:
    data = None
    for fpath in Path(LABELS_ROOT).glob(f'paritcipant_{participant.to_num()}_test_{test.to_num()}.npy'):
        with fpath.open('rb') as f:
            data = np.load(f, allow_pickle=True)
            logging.debug(f'Loaded window {fpath.name}. shape {data.shape}')

    assert data is not None, f'No data found for {participant} and {test}'
    assert len(
        list(Path(SIGNAL_FEATURES_ROOT).glob(f'paritcipant_{participant.to_num()}_test_{test.to_num()}.npy'))) == 1, \
        (f'Found more than'
         f' a single feature'
         f' file for '
         f'{participant} and {test}')
    return data


def config_logger(args, logs_dir = LOGS_DIR):
    logger = logging.getLogger(args.app_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(args.log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logs_dir / f'train_{args.app_name}_{time.asctime()}.log')
    file_handler.setLevel(args.log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
