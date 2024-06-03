import argparse
import logging
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from common.types_defined import Participant, DayT1T2
from common.folder_paths import (VALID_USER_WINDOWS_ROOT)
from common.utils import config_logger
from datasets.not_used.augmented_user_windows_dataset import AugmentedUserWindowsDataset
from datasets.not_used.dataset_composition import DatasetComposition
from models.emg_key_press_classifier import EmgKeyPressClassifier
from train_utils import get_optimizer, train, config_wandb, get_sweep_config, init_sweep, start_sweep, load_checkpoint


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

    parser.add_argument("--data-folder-name", type=str, default='valid_user_windows')

    parser.add_argument("--dataset-name", type=str, default=f"signal_windows",
                        choices=['signal_filtered', 'signal_windows', 'signal_features'],
                        help="Type of preprocessing")

    parser.add_argument("--app-name", type=str,
                        default=f"train_on_all_but_P14_eval_on_him")

    parser.add_argument('--run-or-sweep', type=str, default='run', choices=['run', 'sweep'],
                        help='A single run or a hyperparameter sweep')

    # Preprocessing filter parameters
    parser.add_argument("--high-band", type=float, default=20.0)
    parser.add_argument("--low-band", type=float, default=450.0)
    parser.add_argument("--low-pass", type=float, default=10.0)
    parser.add_argument("--normalize", type=bool, default=False)

    # Model Parameters
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")
    parser.add_argument("--num-channels", type=int, default=16, help="Number of signal channels")
    parser.add_argument("--encoder-depth", type=int, default=1, help="Number of encoder blocks")
    parser.add_argument("--encoder-activations", type=str, default="lrelu",
                        choices=['relu', 'lrelu'])
    parser.add_argument("--encoder-pools", type=str, default='max_pool', choices=['avg_pool', 'max_pool'])

    # Train Hyperparameter
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples in train batch")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Optimizer weight decay parameters")
    parser.add_argument("--momentum", type=float, default="0.9", help="Optimizer momentum parameter")
    parser.add_argument("--use-cuda", type=bool, default=True, help='Use GPU. Use cpu if not')
    parser.add_argument("--saved-models-path", type=str, default=f'{str(Path.home())}/saved_models/KeyPressEMG',
                        help='Train model in a federated_learning manner before fine tuning')
    parser.add_argument("--checkpoint", type=str, default='_epoch_7_acc_0.08100289296046287.pth')

    args = parser.parse_args()
    return args


def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        # args.batch_size = config.batch_size
        args.learning_rate = config.learning_rate
        # args.momentum = config.momentum
        # args.weight_decay = config.weight_decay
        # args.normalize = config.normalize
        # args.high_band = config.high_band
        # args.low_band = config.low_band
        # args.low_pass = config.low_pass
        # args.encoder_depth = config.encoder_depth
        # args.encoder_activations = config.encoder_activations
        # args.encoder_pools = config.encoder_pools

        wandb.run.name = f'depth_{args.encoder_depth}_lr_{args.learning_rate}'

        # wandb.run.name = (f'depth_{args.encoder_depth}_bsz_{args.batch_size}_lr_{args.learning_rate}'
        #                   f'_bp_[{args.high_band}, {args.low_band}]_lp_{args.low_pass}')
        # train_on_user_first_day_eval_same_user_second_day_using_network_raw_data(args)
        train_on_all_but_one_eval_on_him(cli_args, eval_participant=Participant.P15)


def train_on_all_but_one_eval_on_him(args, eval_participant: Participant):
    train_participants = [p for p in Participant if p != eval_participant]
    train_pt = [(p, t) for p in train_participants for t in DayT1T2]
    eval_pt = [(eval_participant, t) for t in DayT1T2]
    train_base(args, train_pt_pairs=train_pt, eval_pt_pairs=eval_pt)


def train_on_user_first_day_eval_same_user_second_day_using_network_raw_data(args):
    for p in Participant:
        train_base(args, train_pt_pairs=[(p, DayT1T2.T1)], eval_pt_pairs=[(p, DayT1T2.T2)], prefix=p.value)


def train_base(args, train_pt_pairs: list[tuple[Participant, DayT1T2]],
               eval_pt_pairs: list[tuple[Participant, DayT1T2]], prefix: str = ''):
    logger = logging.getLogger(args.app_name)

    num_channels = args.num_channels
    num_classes = args.num_classes
    depth = args.encoder_depth
    encoder_sizes = [num_channels * 2 ** (i - 1) for i in range(1, depth + 1)]
    decoder_sizes = list(reversed(encoder_sizes))
    logger.info(f'encoder_sizes: {encoder_sizes}')
    logger.info(f'decoder_sizes: {decoder_sizes}')

    user_input_root = Path(args.data_path) / args.data_folder_name if args.data_folder_name else VALID_USER_WINDOWS_ROOT

    train_set = DatasetComposition(datasets=[AugmentedUserWindowsDataset(root=user_input_root,
                                                                         participant=p,
                                                                         test_day=t,
                                                                         high_band=args.high_band,
                                                                         low_band=args.low_band,
                                                                         low_pass=args.low_pass,
                                                                         apply_augmentation=True,
                                                                         apply_filter=True,
                                                                         apply_rectification=True,
                                                                         apply_dc_remove=True,
                                                                         apply_envelope=True,
                                                                         apply_normalize=True) for (p, t) in
                                             train_pt_pairs])

    eval_set = DatasetComposition(datasets=[AugmentedUserWindowsDataset(root=user_input_root,
                                                                        participant=p,
                                                                        test_day=t,
                                                                        high_band=args.high_band,
                                                                        low_band=args.low_band,
                                                                        low_pass=args.low_pass,
                                                                        apply_augmentation=False,
                                                                        apply_filter=True,
                                                                        apply_rectification=True,
                                                                        apply_dc_remove=True,
                                                                        apply_normalize=True) for (p, t) in
                                            eval_pt_pairs])

    model = EmgKeyPressClassifier(in_c=num_channels,
                                  enc_sizes=encoder_sizes,
                                  dec_sizes=decoder_sizes,
                                  n_classes=num_classes,
                                  activation=args.encoder_activations,
                                  pool=args.encoder_pools)

    num_params = sum([p.numel() for p in model.parameters()])
    logger.info(f'Number of model parameters: {num_params}')

    optimizer = get_optimizer(args, model)

    if args.checkpoint:
        logger.info(f'loading checkpiont {args.checkpoint}')
        checkpoint = load_checkpoint(args)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Continue train from epoch {checkpoint["epoch"]}')
        # logger.info(f'Loaded optimizer state {checkpoint["optimizer_state_dict"]}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    acc, history = train(args, model, train_loader, eval_loader, optimizer, prefix)

    logger.info(f'{prefix} Train Finished: Accuracy score: {acc} history: {history}')


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Train Key Press EMG")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    if cli_args.run_or_sweep == 'run':
        wandb_run = config_wandb(cli_args)
        logger.info(f'wandb run: {wandb_run}')

        # train_on_some_eval_on_other(cli_args)
        # train_on_user_eval_on_same_data_use_svm(cli_args)
        # train_on_user_eval_on_same_data(cli_args)
        # train_on_user_first_day_eval_same_user_second_day_using_svm(cli_args)
        # train_on_user_first_day_eval_same_user_second_day_using_network(cli_args)
        # train_on_user_first_day_eval_same_user_second_day_using_network_raw_data(cli_args)
        train_on_all_but_one_eval_on_him(cli_args, eval_participant=Participant.P15)
    else:
        assert cli_args.run_or_sweep == 'sweep', f'Should be run or sweep'

        # hyperparameter sweep
        sweep_config = get_sweep_config()
        logger.info(f'sweep {sweep_config}')
        sweep_id = init_sweep(sweep_config)
        f_sweep = partial(sweep_train, sweep_id=sweep_id, args=cli_args)
        wandb.agent(sweep_id=sweep_id, function=f_sweep)
        start_sweep(sweep_id, f_sweep)
