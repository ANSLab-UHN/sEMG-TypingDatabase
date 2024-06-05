import argparse
from keypressemg.common.folder_paths import DATA_ROOT
import keypressemg.fl_trainers.trainer_FedAvg_split_day
from keypressemg.common.utils import config_logger
from keypressemg.sweeps.sweep_utils import sweep
from keypressemg.fl_trainers.utils import str2bool


def get_command_line_arguments(parser):
    #############################
    #       Data args        #
    #############################
    parser.add_argument("--data-path", type=str,
                        default=f'{DATA_ROOT.as_posix()}/valid_user_features',
                        help="dir path for datafolder")
    parser.add_argument("--app-name", type=str, default=f"sweep_Fed_Avg_split_day")
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inner-steps", type=int, default=10, help="number of inner steps")
    parser.add_argument("--num-client-agg", type=int, default=5, help="number of cleints per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--eval-every", type=int, default=20, help="eval every X selected steps")
    parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Sweep FedAvg Split Day Train Key Press EMG MLP")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    # hyperparameter sweep
    sweep_configuration = {
        "name": "sweep_FedAvg_split_day_key_press_emg",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "eval_acc"},
        "parameters": {
            "lr": {"values": [0.1]},
            "seed": {"values": [42]},
            "inner_steps": {"values": [20, 40]},
            "wd": {"values": [0.0001]},
            "num_steps": {"values": [600]},
            "num_client_agg": {"values": [10]}
        },
    }
    sweep(sweep_config=sweep_configuration, args=cli_args,
          train_fn=keypressemg.fl_trainers.trainer_FedAvg_split_day.train)
