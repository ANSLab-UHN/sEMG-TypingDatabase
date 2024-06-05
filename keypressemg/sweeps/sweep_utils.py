import logging
from functools import partial
import wandb
from keypressemg.fl_trainers.utils import set_seed


def sweep_train(sweep_id, args, train_fn, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        set_seed(config.seed)

        for k, v in config.items():
            if k in args:
                setattr(args, k, v)

        wandb.run.name = '_'.join([f'{k}_{v}' for k, v in config.items()])
        train_fn(args)


def init_sweep(config):
    sweep_id = wandb.sweep(sweep=config, project="key_press_emg_toronto")
    return sweep_id


def start_sweep(sweep_id, f_sweep):
    wandb.agent(sweep_id=sweep_id, function=f_sweep)


def sweep(sweep_config, args, train_fn):
    logger = logging.getLogger(args.app_name)
    logger.info(f'sweep {sweep_config}')
    sweep_id = init_sweep(sweep_config)
    f_sweep = partial(sweep_train, sweep_id=sweep_id, args=args, train_fn=train_fn)
    wandb.agent(sweep_id=sweep_id, function=f_sweep)
    start_sweep(sweep_id, f_sweep)
