import logging

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common.folder_paths import VALID_NORMALIZED_FEATURES_ROOT
from common.types_defined import Participant, DayT1T2
from datasets.split_same_day_dataset import SplitDayDataset
from models.not_used.factory import get_model
from train_utils import get_optimizer, train


def train_on_user_eval_on_same_data(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    history_dict: dict[str, list[float]] = {}

    for p in Participant:
        for t in DayT1T2:

            train_participants = [p]
            train_experiment = t
            logger.info(f'Train Participants: {train_participants} experiment {train_experiment}')
            eval_participants = [p]
            eval_experiment = t
            logger.info(f'Eval participants: {eval_participants} experiment {eval_experiment}')

            model = get_model(args)
            optimizer = get_optimizer(args, model)

            #****
            train_set = SplitDayDataset(participant=train_participants[0], test=train_experiment,
                                        is_train=True)
            eval_set = SplitDayDataset(participant=eval_participants[0], test=eval_experiment,
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
            logger.info(f'data average sum = {data.sum(1).mean(0)}')

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            logger.info(f'data scaled shape {data_scaled.shape}')
            logger.info(f'data scaled average sum = {data_scaled.sum(1).mean(0)}')

            X_train = torch.from_numpy(data_scaled[:num_samples_train])
            X_test = torch.from_numpy(data_scaled[-num_samples_test:])
            logger.info(f'Scaled X_train shape {X_train.shape} y_train shape {y_train.shape}')
            logger.info(f'Scaled X_test shape {X_test.shape} y_test shape {y_test.shape}')
            #
            user_str = f'{train_participants[0].value}_{train_experiment.value}'
            torch.save(X_train, VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_X_train')
            torch.save(y_train, VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_y_train')

            torch.save(X_test, VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_X_test')
            torch.save(y_test, VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_y_test')

            # X_train = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_X_train')
            # y_train = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_y_train')
            #
            # X_test = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_X_test')
            # y_test = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str}_y_test')
            #****

            logger.info(f'Scaled X_train shape {X_train.shape} y_train shape {y_train.shape}')
            logger.info(f'Scaled X_test shape {X_test.shape} y_test shape {y_test.shape}')

            train_loader = DataLoader(TensorDataset(X_train.float(), y_train.reshape(-1, 1).long()),
                                      batch_size=args.batch_size, shuffle=True)
            eval_loader = DataLoader(TensorDataset(X_test.float(), y_test.reshape(-1, 1).long()),
                                     batch_size=args.batch_size, shuffle=False)

            acc, history = train(args, model, train_loader, eval_loader, optimizer)
            acc_dict[user_str] = acc
            history_dict[user_str] = history
            logger.info(f'{user_str} train evaluation acc: {acc}')
    logger.info('Final results')
    logger.info(f'Accuracy score\n{acc_dict}')
    for u in acc_dict.keys():
        logger.info(f'{u} Accuracy score\n{acc_dict[u]}')
        logger.info(f'{u} History\n{history_dict[u]}')
