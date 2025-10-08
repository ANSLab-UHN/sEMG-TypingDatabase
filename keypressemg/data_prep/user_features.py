import argparse
import logging
from pathlib import Path
import numpy as np
from keypressemg.common.folder_paths import VALID_USER_FEATURES_ROOT, VALID_FEATURES_ROOT, DATA_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2, KeyPress
from keypressemg.common.utils import config_logger


def create_user_features(args):
    logger = logging.getLogger(args.app_name)
    logger.info(f'Args: {args}')

    features_path = Path(args.data_path) / args.features_folder_name
    assert features_path.exists(), f'{features_path} does not exist'

    user_features_path = Path(args.data_path) / args.user_features_folder_name
    if not user_features_path.exists():
        user_features_path.mkdir()

    for p in Participant:

        for t in DayT1T2:
            logger.info(f'Create features  for participant {p} day {t}')

            user_str = f'{p.value}_{t.value}'

            arrays = []
            labels = []
            for fpath in features_path.glob(f'**/{user_str}*.npy'):
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

            with open(user_features_path.joinpath(user_str + '_X.npy'), 'wb') as f:
                np.save(f, user_features)
            with open(user_features_path.joinpath(user_str + '_y.npy'), 'wb') as f:
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
    parser.add_argument("--data-path", type=str, default=DATA_ROOT.as_posix(),
                        help="dir path for datafolder")

    parser.add_argument('--log_level', default='DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')

    parser.add_argument("--features-folder-name", type=str, default=VALID_FEATURES_ROOT.name)

    parser.add_argument("--user-features-folder-name", type=str, default=VALID_USER_FEATURES_ROOT.name)

    parser.add_argument("--app-name", type=str, default=f"User_Features")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Dataset creation - user features")
    cli_args = get_command_line_arguments(argument_parser)


    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    create_user_features(cli_args)
    logger.info("Done Creating User Features")
