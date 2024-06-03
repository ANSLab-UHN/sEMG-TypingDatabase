import logging

import numpy as np
import torch
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common.folder_paths import VALID_NORMALIZED_FEATURES_ROOT, VALID_USER_WINDOWS_ROOT
from common.types_defined import Participant, DayT1T2
from datasets.not_used.in_memory_dataset import load_participant_list_experiments, load_participant_experiments, \
    scale_all_channels
from datasets.split_same_day_dataset import SplitDayDataset


def train_on_user_eval_on_same_data_use_svm(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    f1_dict: dict[str, float] = {}

    for p in Participant:
        for t in DayT1T2:

            train_participants = [p]
            train_experiment = t
            logger.info(f'Train Participants: {train_participants} experiment {train_experiment}')
            eval_participants = [p]
            eval_experiment = t
            logger.info(f'Eval participants: {eval_participants} experiment {eval_experiment}')
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

            X_train = data_scaled[:num_samples_train]
            X_test = data_scaled[-num_samples_test:]

            logger.info(f'after scale X_train shape {X_train.shape} y_train shape {y_train.shape}')
            logger.info(f'after scale X_test shape {X_test.shape} y_test shape {y_test.shape}')

            svm_model = svm.SVC(kernel='rbf')

            logger.info('SVM fit')
            svm_model.fit(X_train, y_train.reshape(-1))

            logger.info('svm predict')
            y_pred = svm_model.predict(X_test)

            acc = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, average='micro')

            logger.info(f'{train_participants[0].value}_{train_experiment.value} acc {acc}')
            logger.info(f'{train_participants[0].value}_{train_experiment.value} f1 {f1}')

            acc_dict[f'{train_participants[0].value}_{train_experiment.value}'] = acc
            f1_dict[f'{train_participants[0].value}_{train_experiment.value}'] = f1

    logger.info('Final results')
    logger.info(f'Accuracy score\n{acc_dict}')
    logger.info(f'F1 score\n{f1_dict}')


def train_on_user_first_day_eval_same_user_second_day_using_svm(args):
    logger = logging.getLogger(args.app_name)

    acc_dict: dict[str, float] = {}
    f1_dict: dict[str, float] = {}

    for p in Participant:
        train_participants = [p]
        train_experiment = DayT1T2.T1
        logger.info(f'Train Participants: {train_participants} experiment {train_experiment}')
        eval_participants = [p]
        eval_experiment = DayT1T2.T2
        logger.info(f'Eval participants: {eval_participants} experiment {eval_experiment}')

        user_str_first_day = f'{train_participants[0].value}_{DayT1T2.T1.value}'
        user_str_second_day = f'{train_participants[0].value}_{DayT1T2.T2.value}'

        X_train_first_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_first_day}_X_train')
        y_train_first_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_first_day}_y_train')
        X_train_second_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_second_day}_X_train')
        y_train_second_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_second_day}_y_train')

        X_test_first_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_first_day}_X_test')
        y_test_first_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_first_day}_y_test')
        X_test_second_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_second_day}_X_test')
        y_test_second_day = torch.load(VALID_NORMALIZED_FEATURES_ROOT / f'{user_str_second_day}_y_test')

        X_first_day = torch.cat([X_train_first_day, X_test_first_day], dim=0)
        X_second_day = torch.cat([X_train_second_day, X_test_second_day], dim=0)
        num_samples_first_day = X_first_day.shape[0]
        num_samples_second_day = X_second_day.shape[0]

        y_first_day = torch.cat([y_train_first_day, y_test_first_day], dim=0)
        y_second_day = torch.cat([y_train_second_day, y_test_second_day], dim=0)

        assert num_samples_first_day == y_first_day.shape[0]
        assert num_samples_second_day == y_second_day.shape[0]

        data = torch.cat([X_first_day, X_second_day], dim=0)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logger.info(f'data scaled shape {data_scaled.shape}')
        logger.info(f'data scaled average sum = {data_scaled.sum(1).mean(0)}')

        X_train = data_scaled[:num_samples_first_day]
        X_test = data_scaled[-num_samples_second_day:]

        y_train = y_first_day
        y_test = y_second_day

        logger.info(f'after scale X_train shape {X_train.shape} y_train shape {y_train.shape}')
        logger.info(f'after scale X_test shape {X_test.shape} y_test shape {y_test.shape}')

        svm_model = svm.SVC(kernel='rbf')

        logger.info('SVM fit')
        svm_model.fit(X_train, y_train.reshape(-1))

        logger.info('svm predict')
        y_pred = svm_model.predict(X_test)

        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='micro')

        logger.info(f'{train_participants[0].value} acc {acc}')
        logger.info(f'{train_participants[0].value} f1 {f1}')

        acc_dict[f'{train_participants[0].value}'] = acc
        f1_dict[f'{train_participants[0].value}'] = f1

    logger.info('Final results')
    logger.info(f'Accuracy score\n{acc_dict}')
    logger.info(f'F1 score\n{f1_dict}')


def train_lopo_svm(args, eval_participant):
    train_participants = [p for p in Participant if p != eval_participant]

    logger = logging.getLogger(args.app_name)

    X_train, y_train = load_participant_list_experiments(VALID_USER_WINDOWS_ROOT, train_participants)
    X_test, y_test = load_participant_experiments(VALID_USER_WINDOWS_ROOT, eval_participant)

    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]

    assert num_samples_train == y_train.shape[0], f'{num_samples_train} != {y_train.shape[0]}'
    assert num_samples_test == y_test.shape[0], f'{num_samples_test} != {y_test.shape[0]}'

    X = np.concatenate((X_train, X_test), axis=0)

    data_scaled = scale_all_channels(StandardScaler(), X)
    logger.info(f'data scaled shape {data_scaled.shape}')


    X_train = data_scaled[:num_samples_train]
    X_test = data_scaled[-num_samples_test:]

    svm_model = svm.SVC(kernel='rbf')

    logger.info('SVM fit')
    svm_model.fit(X_train.reshape(num_samples_train, -1), y_train.reshape(-1))

    logger.info('svm predict')
    y_pred = svm_model.predict(X_test.reshape(num_samples_test, -1))

    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='micro')
    cm = metrics.confusion_matrix(y_test, y_pred)

    logger.info(f'{train_participants[0].value} acc {acc}')
    logger.info(f'{train_participants[0].value} f1 {f1}')

    return acc, f1, cm
