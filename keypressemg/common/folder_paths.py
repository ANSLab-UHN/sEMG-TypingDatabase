from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

LOGS_DIR = PROJECT_DIR / 'log'
DATA_ROOT = Path(PROJECT_DIR).parent / 'CleanData'
RAW_SIGNAL_ROOT = DATA_ROOT / 'signal_filtered'
SIGNAL_WINDOWS_ROOT = DATA_ROOT / 'signal_windows'
SIGNAL_FEATURES_ROOT = DATA_ROOT / 'signal_features'
LABELS_ROOT = DATA_ROOT / 'labels'
VALID_EXPERIMENTS_ROOT = DATA_ROOT / 'valid_experiments'
VALID_WINDOWS_ROOT = DATA_ROOT / 'valid_windows'
VALID_USER_WINDOWS_ROOT = DATA_ROOT / 'valid_user_windows'
VALID_USER_FEATURES_ROOT = DATA_ROOT / 'valid_user_features'
VALID_FEATURES_ROOT = DATA_ROOT / 'valid_features'
VALID_NORMALIZED_FEATURES_ROOT = DATA_ROOT / 'valid_normalized_feature_root'
VALID_NORMALIZED_WINDOWS_ROOT = DATA_ROOT / 'valid_normalized_windows'
