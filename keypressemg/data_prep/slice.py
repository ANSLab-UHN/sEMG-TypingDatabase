import argparse
import logging
from pathlib import Path
from keypressemg.common.folder_paths import VALID_WINDOWS_ROOT, DATA_ROOT, VALID_EXPERIMENTS_ROOT
from keypressemg.common.utils import config_logger
from keypressemg.data_prep.experiments_read.rhd_data import RHDDataFileLoader
from keypressemg.data_prep.window_prepare.window_slice import RHDWindowSlicer


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

    parser.add_argument("--valid-folder-name", type=str, default=VALID_EXPERIMENTS_ROOT.name)

    parser.add_argument("--window-folder-name", type=str, default=VALID_WINDOWS_ROOT.name)

    parser.add_argument("--app-name", type=str, default=f"Slice_To_Windows")

    parser.add_argument("--window-length-seconds", type=float, default=0.2)

    args = parser.parse_args()
    return args


def main(args):
    logger = logging.getLogger(args.app_name)
    valid_folder = Path(args.data_path) / args.valid_folder_name
    assert valid_folder.exists(), f'{valid_folder} does not exist'
    assert valid_folder.is_dir(), f'{valid_folder} is not a directory'

    window_folder = Path(args.data_path) / args.window_folder_name
    if not window_folder.exists():
        logger.info(f'{window_folder} does not exist. Creating...')
        window_folder.mkdir()
    assert window_folder.exists(), f'{window_folder} does not exist'
    assert window_folder.is_dir(), f'{window_folder} is not a directory'

    for valid_rhd in valid_folder.iterdir():
        rhd_valid_file_loader = RHDDataFileLoader(filepath=valid_rhd)
        rhd_valid_file_loader.load()
        rhd_data = rhd_valid_file_loader.data
        window_slicer = RHDWindowSlicer(rhd_data, win_length_seconds=cli_args.window_length_seconds)
        logger.info(f'Slicing: {rhd_data.name}')
        window_slicer.slice()
        window_slicer.save_windows(Path(args.data_path) / args.window_folder_name)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Slice To Windows")
    cli_args = get_command_line_arguments(argument_parser)

    DATA_ROOT = Path(cli_args.data_path)
    VALID_EXPERIMENTS_ROOT = DATA_ROOT / cli_args.valid_folder_name
    VALID_WINDOWS_ROOT = DATA_ROOT / cli_args.window_folder_name

    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    main(cli_args)
    logger.info("Done Slicing To Windows")
