from pathlib import Path

from torch.utils.data import DataLoader
from datasets.split_same_day_dataset import SplitDayDataset
from datasets.split_between_days_dataset import SplitBetweenDaysDataset
from common.types_defined import Participant, DayT1T2


def get_loader(args,
               root: Path,
               participant: Participant, is_train: bool = True):

    raise Exception('Factory logic Not consistent with datasets')

    dataset_name = args.dataset_name
    assert dataset_name in ['signal_windows', 'signal_features'], (f'Dataset {dataset_name} not supported.'
                                                                   f' Expected windows or features.')
    dataset_ctor = SplitBetweenDaysDataset if dataset_name == 'signal_windows' else SplitDayDataset
    dataset = dataset_ctor(root=root, participant=participant, test=DayT1T2.T1 if is_train else DayT1T2.T2)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=is_train, num_workers=2)

