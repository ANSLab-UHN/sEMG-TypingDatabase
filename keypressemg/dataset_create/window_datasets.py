import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from common.folder_paths import VALID_NORMALIZED_WINDOWS_ROOT, VALID_WINDOWS_ROOT, VALID_USER_WINDOWS_ROOT
from common.types_defined import Participant, DayT1T2
from datasets.split_between_days_dataset import SplitBetweenDaysDataset
from common.utils import config_logger


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
    parser.add_argument("--target-root", type=str, default=VALID_USER_WINDOWS_ROOT, help="Root path for target dataset")
    parser.add_argument("--app-name", type=str, default=f"create_user_windows")

    args = parser.parse_args()
    return args


def create_normalized_windows(args):
    logger = config_logger(args)
    logger.info(f'Args: {args}')

    for p in Participant:

        train_participants = [p]
        train_experiment = DayT1T2.T1
        logger.info(f'Train Participants: {train_participants} experiment {train_experiment}')
        eval_participants = [p]
        eval_experiment = DayT1T2.T2
        logger.info(f'Eval participants: {eval_participants} experiment {eval_experiment}')

        user_str = f'{train_participants[0].value}'

        train_set = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                            participant=train_participants[0],
                                            is_train=True)
        eval_set = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                           participant=eval_participants[0],
                                           is_train=False)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False)
        X_train, y_train = torch.tensor([]), torch.tensor([])
        logger.info(f'construct train set')

        for data, target in tqdm(train_loader):
            X_train = torch.cat((X_train, data), dim=0)
            y_train = torch.cat((y_train, target), dim=0)

        num_samples_train = X_train.shape[0]
        assert num_samples_train == y_train.shape[0], (f'Expected same number of data samples and targets.'
                                                       f' Got {num_samples_train} data samples'
                                                       f' and {y_train.shape[0]} targets')
        logger.info(f'X_train shape {X_train.shape} y_train shape {y_train.shape}')
        logger.info(f'construct eval set')
        X_test, y_test = torch.tensor([]), torch.tensor([])
        for data, target in tqdm(eval_loader):
            X_test = torch.cat((X_test, data), dim=0)
            y_test = torch.cat((y_test, target), dim=0)

        num_samples_test = X_test.shape[0]
        assert num_samples_test == y_test.shape[0], (f'Expected same number of data samples and targets.'
                                                     f' Got {num_samples_test} data samples'
                                                     f' and {y_test.shape[0]} targets')
        logger.info(f'X_test shape {X_test.shape} y_test shape {y_test.shape}')

        data = torch.cat([X_train, X_test], dim=0)
        assert data.shape[0] == num_samples_train + num_samples_test
        logger.info(f'data concat for scaling. shape {data.shape}')

        data_scaled = (data - data.mean(0)) / data.std(0)

        logger.info(f'data scaled shape {data_scaled.shape}')

        X_train = data_scaled[:num_samples_train]
        X_test = data_scaled[-num_samples_test:]

        logger.info(f'Scaled X_train shape {X_train.shape} y_train shape {y_train.shape}')
        logger.info(f'Scaled X_test shape {X_test.shape} y_test shape {y_test.shape}')

        torch.save(X_train, VALID_NORMALIZED_WINDOWS_ROOT / f'{user_str}_X_train.pt')
        torch.save(y_train, VALID_NORMALIZED_WINDOWS_ROOT / f'{user_str}_y_train.pt')

        torch.save(X_test, VALID_NORMALIZED_WINDOWS_ROOT / f'{user_str}_X_test.pt')
        torch.save(y_test, VALID_NORMALIZED_WINDOWS_ROOT / f'{user_str}_y_test.pt')


def create_user_windows(args):
    logger = config_logger(args)
    logger.info(f'Args: {args}')

    for p in Participant:

        logger.info(f'Create windows for participant {p}')

        user_str = f'{p.value}'

        t1_set = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                         participant=p,
                                         is_train=True)
        t2_set = SplitBetweenDaysDataset(root=VALID_WINDOWS_ROOT,
                                         participant=p,
                                         is_train=False)

        t1_loader = torch.utils.data.DataLoader(t1_set, batch_size=1, shuffle=False)
        t2_loader = torch.utils.data.DataLoader(t2_set, batch_size=1, shuffle=False)

        X_t1, y_t1 = torch.tensor([]), torch.tensor([])
        logger.info(f'construct t1 set')

        for data, target in tqdm(t1_loader):
            X_t1 = torch.cat((X_t1, data), dim=0)
            y_t1 = torch.cat((y_t1, target), dim=0)

        num_samples_t1 = X_t1.shape[0]
        assert num_samples_t1 == y_t1.shape[0], (f'Expected same number of data samples and targets.'
                                                 f' Got {num_samples_t1} data samples'
                                                 f' and {y_t1.shape[0]} targets')
        logger.info(f'X_t1 shape {X_t1.shape} y_t1 shape {y_t1.shape}')
        logger.info(f'construct t2 set')
        X_t2, y_t2 = torch.tensor([]), torch.tensor([])
        for data, target in tqdm(t2_loader):
            X_t2 = torch.cat((X_t2, data), dim=0)
            y_t2 = torch.cat((y_t2, target), dim=0)

        num_samples_t2 = X_t2.shape[0]
        assert num_samples_t2 == y_t2.shape[0], (f'Expected same number of data samples and targets.'
                                                 f' Got {num_samples_t2} data samples'
                                                 f' and {y_t2.shape[0]} targets')
        logger.info(f'X_t2 shape {X_t2.shape} y_t2 shape {y_t2.shape}')

        target_root: Path = Path(args.target_root)
        assert target_root.exists(), f'{target_root} does not exist'
        assert target_root.is_dir(), f'{target_root} is not a directory'

        torch.save(X_t1, target_root / f'{user_str}_X_T1.pt')
        torch.save(y_t1, target_root / f'{user_str}_y_T1.pt')

        torch.save(X_t2, target_root / f'{user_str}_X_T2.pt')
        torch.save(y_t2, target_root / f'{user_str}_y_T2.pt')


#
if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Dataset creation")
    cli_args = get_command_line_arguments(argument_parser)

    # create_normalized_windows(cli_args)
    create_user_windows(cli_args)
