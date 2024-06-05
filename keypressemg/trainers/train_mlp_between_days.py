import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from keypressemg.fl_trainers.utils import str2bool
from keypressemg.common.folder_paths import DATA_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2
from keypressemg.common.utils import config_logger
from keypressemg.datasets.split_between_days_dataset import get_split_between_days_dataset
from keypressemg.models.feature_model import FeatureModel
from keypressemg.trainers.utils import train, config_wandb


def train_participant_between_days(args, participant=Participant.P7):
    train_set, eval_set = get_split_between_days_dataset(Path(args.data_path) / args.data_folder_name,
                                                         participant=participant,
                                                         train_day=DayT1T2.T1, scale=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
    model = FeatureModel(cls_layer=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    acc, history = train(args, model, train_loader, eval_loader, optimizer, log_prefix=f'{participant.value}')
    return acc, history


def main(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    history_dict: dict[str, float] = {}

    for p in Participant:
        acc, history = train_participant_between_days(args, p)

        logger.info(f'{p.value} acc {acc}')
        logger.info(f'{p.value} history {history}')

        acc_dict[f'{p.value}'] = acc
        history_dict[f'{p.value}'] = history

        logger.info('Final results')
        logger.info(f'Accuracy score\n{acc_dict}')
        logger.info(f'Acc History \n{history_dict}')


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

    parser.add_argument("--app-name", type=str,
                        default=f"train_mlp_between_days", )

    # Model Parameters
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")

    parser.add_argument("--num-epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--dec_sizes", type=list, default=[96, 26], help="Decoder Layer Sizes")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples in train batch")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0, help="Optimizer weight decay parameters")
    parser.add_argument("--momentum", type=float, default="0.8", help="Optimizer momentum parameter")
    parser.add_argument("--use-cuda", type=bool, default=True, help='Use GPU. Use cpu if not')
    parser.add_argument("--saved-models-path", type=str, default='',
                        help='Train model in a federated_learning manner before fine tuning')
    parser.add_argument('--wandb', type=str2bool, default=False)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Train Key Press EMG MLP Between Days")
    cli_args = get_command_line_arguments(argument_parser)

    DATA_ROOT = Path(cli_args.data_path)
    VALID_USER_FEATURES_ROOT = DATA_ROOT / cli_args.data_folder_name

    logger = config_logger(cli_args, logs_dir=Path(__file__).parent.parent / 'log')
    logger.info(f'Arguments: {cli_args}')

    if cli_args.wandb:
        wandb_run = config_wandb(cli_args)
        logger.info(f'wandb run: {wandb_run}')

    main(cli_args)
