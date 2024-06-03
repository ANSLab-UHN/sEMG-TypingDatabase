from models.feature_model import FeatureModel
from models.not_used.window_model import WindowModel


def get_model(args):
    dataset_name = args.dataset_name
    assert dataset_name in ['signal_windows', 'signal_features'], (f'Dataset {dataset_name} not supported.'
                                                                   f' Expected windows or features.')
    return WindowModel(use_group_norm=False, use_dropout=False) if dataset_name == 'signal_windows' else FeatureModel()
