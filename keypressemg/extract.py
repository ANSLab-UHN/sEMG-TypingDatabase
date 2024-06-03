import argparse
import logging
from pathlib import Path

from common.folder_paths import VALID_FEATURES_ROOT, VALID_WINDOWS_ROOT
from common.utils import config_logger
from window_prepare.feature_window import FeatureWindow


def main(args):
    logger = logging.getLogger(args.app_name)

    for wp in args.window_folder_path.iterdir():
        logger.info(f'extracting features for {wp.as_posix()}')
        fw = FeatureWindow(wp, feature_windows_path=args.features_folder_path)
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
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/GIT/KeypressEMG/CleanData",
                        help="dir path for datafolder")

    parser.add_argument("--features-folder-path", type=Path, default=VALID_FEATURES_ROOT)

    parser.add_argument("--window-folder-path", type=Path, default=VALID_WINDOWS_ROOT)

    parser.add_argument("--app-name", type=str,
                        default=f"Extract_Features")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description="Extract Features")
    cli_args = get_command_line_arguments(argument_parser)
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    main(cli_args)
    logger.info("Done Extracting features")
