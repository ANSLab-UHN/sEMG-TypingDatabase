import argparse
import logging
from pathlib import Path
from keypressemg.common.folder_paths import DATA_ROOT, VALID_FEATURES_ROOT, VALID_WINDOWS_ROOT
from keypressemg.common.utils import config_logger
from keypressemg.data_prep.window_prepare.feature_window import FeatureWindow


def main(args):
    logger = logging.getLogger(args.app_name)

    for wp in VALID_WINDOWS_ROOT.iterdir():
        logger.info(f'extracting features for {wp.as_posix()}')
        fw = FeatureWindow(wp, feature_windows_path=VALID_FEATURES_ROOT)
        fw.load_window()
        fw.extract_window_features()
        fw.save()


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

    parser.add_argument('--log_level', default='DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')

    parser.add_argument("--features-folder-name", type=str, default=VALID_FEATURES_ROOT.name)

    parser.add_argument("--window-folder-name", type=Path, default=VALID_WINDOWS_ROOT.name)

    parser.add_argument("--app-name", type=str,
                        default=f"Extract_Features")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Extract Features")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    DATA_ROOT = Path(cli_args.data_path)
    VALID_FEATURES_ROOT = DATA_ROOT / cli_args.features_folder_name
    VALID_WINDOWS_ROOT = DATA_ROOT / cli_args.window_folder_name
    logger.info(f'Arguments: {cli_args}')

    main(cli_args)
    logger.info("Done Extracting features")
