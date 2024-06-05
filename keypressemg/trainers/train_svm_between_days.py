import argparse
import logging
from pathlib import Path
from sklearn import svm, metrics
from keypressemg.common.folder_paths import DATA_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2
from keypressemg.common.utils import config_logger
from keypressemg.datasets.split_between_days_dataset import get_split_between_days_arrays


def main(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    f1_dict: dict[str, float] = {}

    for p in Participant:
        for t in DayT1T2:
            X_train, y_train, X_test, y_test = get_split_between_days_arrays(
                root=Path.cwd() / args.data_path / args.data_folder_name, participant=p, train_day=t, scale=True)

            num_samples_train = X_train.shape[0]
            assert num_samples_train == y_train.shape[0], (f'Expected same number of data samples and targets.'
                                                           f' Got {num_samples_train} data samples'
                                                           f' and {y_train.shape[0]} targets')
            logger.info(f'X_train shape {X_train.shape} y_train shape {y_train.shape}')

            num_samples_test = X_test.shape[0]
            assert num_samples_test == y_test.shape[0], (f'Expected same number of data samples and targets.'
                                                         f' Got {num_samples_test} data samples'
                                                         f' and {y_test.shape[0]} targets')

            svm_model = svm.SVC(kernel='rbf')

            logger.info('SVM fit')
            svm_model.fit(X_train, y_train.reshape(-1))

            logger.info('svm predict')
            y_pred = svm_model.predict(X_test)

            acc = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, average='micro')

            logger.info(f'{p.value}_{t.value} acc {acc}')
            logger.info(f'{p.value}_{t.value} f1 {f1}')

            acc_dict[f'{p.value}_{t.value}'] = acc
            f1_dict[f'{p.value}_{t.value}'] = f1

        logger.info('Final results')
        logger.info(f'Accuracy score\n{acc_dict}')
        logger.info(f'F1 score\n{f1_dict}')


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

    parser.add_argument("--data-folder-name", type=str, default='valid_user_features')

    parser.add_argument("--app-name", type=str, default=f"train_svm_between_days")

    # Model Parameters
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Train Key Press EMG SVM Between Days")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    main(cli_args)
