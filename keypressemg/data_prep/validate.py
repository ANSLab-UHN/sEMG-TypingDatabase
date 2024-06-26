import argparse
import logging
from pathlib import Path
from keypressemg.common.folder_paths import DATA_ROOT, VALID_EXPERIMENTS_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2, KeyPress
from keypressemg.common.utils import config_logger
from keypressemg.data_prep.experiments_read.experiment import Experiment
from keypressemg.data_prep.experiments_read.experiment_reader import ExperimentReader
from keypressemg.data_prep.experiments_read.rhd_file import RHDFile
from keypressemg.data_prep.experiments_read.rhd_data import RHDDataFileSaver


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

    parser.add_argument("--valid-folder-name", type=str, default=VALID_EXPERIMENTS_ROOT.name)

    parser.add_argument("--app-name", type=str,
                        default=f"Validate_Experiments")

    args = parser.parse_args()
    return args


def main(args):
    logger = logging.getLogger(args.app_name)
    data_folder_path = Path(args.data_path)
    assert data_folder_path.exists(), f'{data_folder_path} does not exist'
    assert data_folder_path.is_dir(), f"{data_folder_path} is not a directory"

    save_dir_path = Path(data_folder_path / args.valid_folder_name)
    if not save_dir_path.exists():
        logger.info(f'{save_dir_path} does not exist. Creating it.')
        save_dir_path.mkdir()
    else:
        logger.info(f'{save_dir_path} already exists. Overwriting it.')

    for p in Participant:
        for t in DayT1T2:
            pt = Experiment(folder_path=data_folder_path,
                            participant=p,
                            test=t)
            logger.info(f'validating {pt.participant} {pt.test}')
            for key in KeyPress:

                logger.info(f'Validating {key}...')

                experiment_reader = ExperimentReader(experiment=pt, key=key)

                logger.info(pt.participant)
                logger.info(pt.test)
                logger.info(key)
                logger.info('validating experiment')

                experiment_reader.validate()

                logger.info(f'Validated Files: {experiment_reader.validated_files}')

                for fp in experiment_reader.validated_files:
                    logger.info(f'reading {fp}')
                    rhd: RHDFile = experiment_reader.read(fp)
                    logger.info(f'data shape {rhd.amplifier_data.shape}')
                    logger.info(f'data timestamps shape {rhd.data_timestamps.shape}')
                    logger.info(f'sampling rate {rhd.sampling_rate}')

                    rhd_data_saver = RHDDataFileSaver(rhd=rhd.rhd_data, save_path=save_dir_path)

                    rhd_data_saver.save()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Validate Recordings")
    cli_args = get_command_line_arguments(argument_parser)
    DATA_ROOT = Path(cli_args.data_path)
    VALID_EXPERIMENTS_ROOT = DATA_ROOT / cli_args.valid_folder_name
    logger = config_logger(cli_args)
    logger.info(f'Arguments: {cli_args}')

    main(cli_args)

    logger.info('Done validating recordings')
