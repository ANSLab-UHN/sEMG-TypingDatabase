import argparse
import logging
from pathlib import Path

import numpy as np

from common.folder_paths import VALID_USER_FEATURES_ROOT, VALID_FEATURES_ROOT
from common.types_defined import Participant, DayT1T2, KeyPress
from common.utils import config_logger


def create_user_features(args):
    logger = logging.getLogger(args.app_name)
    logger.info(f'Args: {args}')

    if not args.user_features_path.exists():
        args.user_features_path.mkdir()

    for p in Participant:

        for t in DayT1T2:
            logger.info(f'Create features  for participant {p} day {t}')

            user_str = f'{p.value}_{t.value}'

            arrays = []
            labels = []
            for fpath in args.features_path.glob(f'**/{user_str}*.npy'):
                label_key = KeyPress.from_str(fpath.name[fpath.name.index('_time_') - 1])
                label = label_key.to_num()
                labels.append(label)
                with open(fpath, 'rb') as f:
                    arr = np.load(f).reshape(1,-1)
                    arrays.append(arr)
            user_features = np.concatenate(arrays, axis=0)
            logger.info(f'User features shape: {user_features.shape}')
            user_labels = np.array(labels)
            logger.info(f'User labels shape: {user_labels.shape}')

            with open(args.user_features_path.joinpath(user_str + '_X.npy'), 'wb') as f:
                np.save(f, user_features)
            with open(args.user_features_path.joinpath(user_str + '_y.npy'), 'wb') as f:
                np.save(f, user_labels)


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

    parser.add_argument("--features-path", type=Path, default=VALID_FEATURES_ROOT)

    parser.add_argument("--user-features-path", type=Path, default=VALID_USER_FEATURES_ROOT)

    parser.add_argument("--app-name", type=str,
                        default=f"User_Features")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Dataset creation - user features")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    create_user_features(cli_args)
    logger.info("Done Creating User Features")
