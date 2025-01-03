import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from keypressemg.fl_trainers.utils import str2bool
from keypressemg.common.folder_paths import DATA_ROOT, PROJECT_DIR, LOGS_DIR
from keypressemg.common.types_defined import Participant, DayT1T2
from keypressemg.common.utils import config_logger
from keypressemg.datasets.split_same_day_dataset import get_same_split_day_datasets
from keypressemg.models.feature_model import FeatureModel
from keypressemg.trainers.utils import train, config_wandb


def main(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    history_dict: dict[str, float] = {}

    for p in Participant:
        for t in DayT1T2:
            train_set, eval_set = get_same_split_day_datasets(Path(args.data_path) / args.data_folder_name, p, t,
                                                              scale=True)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

            model = FeatureModel(cls_layer=True,depth_power=args.depth_power)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

            acc, history = train(args, model, train_loader, eval_loader, optimizer, log_prefix=f'{p.value}_{t.value}')


            logger.info(f'{p.value}_{t.value} acc {acc}')
            logger.info(f'{p.value}_{t.value} history {history}')

            acc_dict[f'{p.value}_{t.value}'] = acc
            history_dict[f'{p.value}_{t.value}'] = history

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
                        default=f"train_mlp_split_day", )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    # Model Parameters
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")
    parser.add_argument("--depth_power", type=int, default=3, help="Determines the network depth")

    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples in train batch")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Optimizer weight decay parameters")
    parser.add_argument("--momentum", type=float, default="0.9", help="Optimizer momentum parameter")
    parser.add_argument("--use-cuda", type=bool, default=True, help='Use GPU. Use cpu if not')
    parser.add_argument("--saved-models-path", type=str, default='',
                        help='Train model in a federated_learning manner before fine tuning')
    parser.add_argument('--wandb', type=str2bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Train Key Press EMG MLP Split Day")
    cli_args = get_command_line_arguments(argument_parser)
    PROJECT_DIR = Path(__file__).parent.parent

    DATA_ROOT = Path(cli_args.data_path)

    LOG_ROOT = PROJECT_DIR / 'logs'

    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    if cli_args.wandb:
        wandb_run = config_wandb(cli_args)
        logger.info(f'wandb run: {wandb_run}')

    main(cli_args)
