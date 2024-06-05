import datetime
import logging
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from keypressemg.common.types_defined import KeyPress
from keypressemg.data_prep.experiments_read.experiment import Experiment
from keypressemg.data_prep.experiments_read.rhd_file import RHDFile
from keypressemg.data_prep.experiments_read.rhd_data import RHDData
from keypressemg.data_prep.experiments_read.utils import get_rhd_file_start_time, get_first_dataframe_row_index_after_base_time, \
    is_first_key_after_index_not_too_long, unique_letters_in_window


class ExperimentReader:
    MINIMUM_KEY_PRESS_NUM: int = 5
    STANDARD_KEY_PRESS_NUM: int = 10
    STANDARD_SPACE_NUM: int = 5
    TRIAL_LENGTH: int = 15
    NUM_CHANNELS: int = 16
    MAX_DELTA_TIME_FOR_KEY_STROKE: datetime.timedelta = datetime.timedelta(days=0, seconds=1)

    class ErrorMessageLevel(Enum):
        WARNING = 'Warning'
        ERROR = 'Error'
        USED = 'Used'

    def __init__(self, experiment: Experiment, key: KeyPress):

        self._err_dataframe = pd.DataFrame(
            {
                'Participant': [],
                'Test': [],
                'Filename': [],
                'Error': [],
                'Warning': [],
                'Used': []
            }
        )
        self._key: KeyPress = key
        self._experiment: Experiment = experiment
        self._key_filepaths: list[Path] = experiment.get_letter_filepaths(key=key)
        self._rhd_file_start_time: dict[Path, datetime.datetime] = {kfp: get_rhd_file_start_time(kfp.stem)
                                                                    for kfp in self._key_filepaths}
        self._rhd_file_first_keylog_index: dict[Path, pd.Index] = {
            kfp: get_first_dataframe_row_index_after_base_time(self._rhd_file_start_time[kfp],
                                                               experiment.key_logs_sorted,
                                                               sorted_df_given=True) for kfp in self._key_filepaths}

        self._file_validated: list[Path] = []
        self._validated_rhd: list[RHDData] = []

        self._trial_timings_absolute: dict[Path, np.ndarray] = {}
        self._key_timings_adjusted: dict[Path, np.ndarray] = {}

        self._data = {kfp: np.array([]) for kfp in self._key_filepaths}
        self._data_timestamps = {kfp: 0 for kfp in self._key_filepaths}
        self._sampling_rate = {kfp: 0 for kfp in self._key_filepaths}
        self._file_ok = {kfp: True for kfp in self._key_filepaths}

    @property
    def key_filepaths(self) -> list[Path]:
        return self._key_filepaths

    @property
    def validated_files(self) -> list[Path]:
        return self._file_validated

    def validate(self):
        for kfp in self._key_filepaths:
            logging.info(f'Validating {kfp}')
            valid = self._validate_file(kfp)
            if valid:
                self._file_validated.append(kfp)
                logging.info(f'File {kfp} validated')
                self._adjust_key_timings(kfp)
            else:
                logging.info(f'File {kfp} not valid')

    def _adjust_key_timings(self, fp: Path) -> None:
        trial_timings_relative = (self._trial_timings_absolute[fp]['Time'] - self._rhd_file_start_time[fp])
        trial_timings_relative = [round(time.total_seconds(), 4) for time in trial_timings_relative]
        lt = self._experiment.lag_timings
        fp_lag = lt.loc[lt['Filename'] == fp.stem, 'Lag']
        logging.debug(f'_adjust_key_timings lag {fp_lag}')
        self._key_timings_adjusted[fp] = trial_timings_relative + np.full(len(trial_timings_relative), fp_lag)
        logging.debug(f'key timings adjusted for {fp}: {self._key_timings_adjusted[fp]}')

    def _validate_file(self, fp: Path) -> bool:
        valid_first_space = self._check_valid_first_space(fp)
        logging.debug(f'{fp} first space valid: {valid_first_space}')
        valid_key_log_sequence = False
        if valid_first_space:
            valid_key_log_sequence = self._check_valid_key_log_sequence(fp)
            logging.debug(f'{fp} key log sequence valid: {valid_key_log_sequence}')
        file_valid = valid_first_space and valid_key_log_sequence
        logging.debug(f'{fp} valid: {file_valid}')
        return file_valid

    def _check_valid_first_space(self, fp: Path) -> bool:
        # Go to closest keylog time to start and look for
        # the next space bar press to indicate the start of a trial.
        # If no space press is found label as a bad trial
        return is_first_key_after_index_not_too_long(df=self._experiment.key_logs,
                                                     key=KeyPress.SPACE,
                                                     pd_index=self._rhd_file_first_keylog_index[fp],
                                                     too_long_time=datetime.timedelta(days=0, seconds=1))

    def _append_error_message(self,
                              fp: Path,
                              level: ErrorMessageLevel,
                              message: str) -> None:

        logging.warning(f'Participant {self._experiment.participant.to_num()}, '
                        f'Test: {self._experiment.test.to_num()},'
                        f' Filename: {fp.stem},'
                        f' {level.value}: {message}')

        self._err_dataframe = pd.concat([self._err_dataframe, pd.DataFrame(
            {'Participant': [self._experiment.participant.to_num()],
             'Test': [self._experiment.test.to_num()],
             'Filename': [fp.stem],
             level.value: message})])

    def _check_valid_key_log_sequence(self, file_path: Path) -> bool:

        first_key_stroke_index: pd.Index = self._rhd_file_first_keylog_index[file_path]
        experiment: Experiment = self._experiment

        logging.debug(f'_check_valid_key_log_sequence({file_path}) first_key_stroke_index {first_key_stroke_index} ')

        i = 0
        BAD_FILE, keylog_clear = False, False
        # Check that the right sequence of key logs is present after the initial time.
        while not (BAD_FILE or keylog_clear):
            window_length = ExperimentReader.TRIAL_LENGTH - 1 + i
            logging.debug(f'in while loop i = {i} window_length = {window_length}')
            trial_window, unique_items, counts = unique_letters_in_window(window_start_index=first_key_stroke_index,
                                                                          window_length=window_length,
                                                                          keylog=experiment.key_logs)
            logging.debug(f'trial_window {trial_window}\nunique_items {unique_items}\ncounts {counts}\n')
            # If there is no letter found in the next TRIAL_LENGTH key presses,
            # it's a bad file and discarded.
            if self._key.value not in unique_items:
                BAD_FILE = True
                self._append_error_message(file_path,
                                           ExperimentReader.ErrorMessageLevel.ERROR,
                                           'No correct keypress found at that time')
            else:
                # Otherwise count the number of spaces and key presses
                # present in the current window.
                logging.debug(f'key {self._key.value} in unique items: {unique_items}')
                space_count = counts[np.where(unique_items == 'Key.space')[0][0]]
                key_count = counts[np.where(unique_items == self._key.value)[0][0]]

                logging.debug(f'space count {space_count}')
                logging.debug(f'key count {key_count}')

                assert isinstance(space_count, np.int_), 'Expected space count to be an integer'
                assert isinstance(key_count, np.int_), 'Expected key count to be an integer'

                BAD_FILE, i, keylog_clear = self._check_space_vs_letter_count(
                    file_path,
                    i,
                    int(key_count),
                    int(space_count),
                    trial_window)

                logging.debug(f'_check_space_vs_letter_count returned with '
                              f'BAD_FILE={BAD_FILE} i={i} keylog_clear={keylog_clear}')

        return not BAD_FILE

    def _check_space_vs_letter_count(self,
                                     file_path: Path,
                                     i: int,
                                     key_count: int,
                                     space_count: int,
                                     trial_window):
        # If it is STANDARD_SPACE_NUM and STANDARD_KEY_PRESS_NUM space and key presses respectively
        # the keylog is perfect and the timings are extracted
        BAD_FILE, keylog_clear = False, False
        if (i == 0
                and space_count == ExperimentReader.STANDARD_SPACE_NUM
                and key_count == ExperimentReader.STANDARD_KEY_PRESS_NUM):
            logging.debug(f'_check_space_vs_letter_count({file_path}) perfect keylog')
            keylog_clear = True
            self._trial_timings_absolute[file_path] = trial_window[
                np.logical_or(trial_window['Letter'] == self._key.value, trial_window['Letter'] == 'Key.space')]
        # If we have STANDARD_KEY_PRESS_NUM  key presses then we are still satisfied
        # despite the keylog not being perfect.
        # This is noted in the error file.
        elif key_count == ExperimentReader.STANDARD_KEY_PRESS_NUM:
            logging.debug(f'_check_space_vs_letter_count({file_path})  not perfect keylog but accepted')
            keylog_clear = True
            self._trial_timings_absolute[file_path] = trial_window[
                np.logical_or(trial_window['Letter'] == self._key.value, trial_window['Letter'] == 'Key.space')]
            self._append_error_message(file_path,
                                       ExperimentReader.ErrorMessageLevel.WARNING,
                                       'Non perfect keylog')
        # If we have not found STANDARD_KEY_PRESS_NUM  key presses look at
        # the next element and check if the next element is the desired letter
        else:
            if key_count < ExperimentReader.STANDARD_KEY_PRESS_NUM:
                if self._is_next_key_equals_letter(self._rhd_file_first_keylog_index[file_path], i):
                    i += 1
                else:
                    # If it is not the desired letter,
                    # as long as we have more than the
                    # RHDReader.MINIMUM_KEY_PRESS_NUM correct key presses,
                    # the file is kept.
                    # Otherwise, it is rejected.
                    BAD_FILE, keylog_clear = self._check_minimum_key_count(file_path,
                                                                           key_count,
                                                                           trial_window)
        return BAD_FILE, i, keylog_clear

    def _check_minimum_key_count(self, file_path: Path,
                                 key_count: int,
                                 trial_window):
        # If it is not the desired letter as long as we have more than the
        # RHDReader.MINIMUM_KEY_PRESS_NUM correct key presses the file is kept.
        # Otherwise, it is rejected.
        BAD_FILE, keylog_clear = False, False
        if key_count >= ExperimentReader.MINIMUM_KEY_PRESS_NUM:
            keylog_clear = True

            self._trial_timings_absolute[file_path] = trial_window[
                np.logical_or(trial_window['Letter'] == self._key.value,
                              trial_window['Letter'] == 'Key.space')]
            self._append_error_message(file_path,
                                       ExperimentReader.ErrorMessageLevel.WARNING,
                                       'Less than 10 keypresses found')
        else:
            BAD_FILE = True
            self._append_error_message(file_path,
                                       ExperimentReader.ErrorMessageLevel.ERROR,
                                       'Less than minimum acceptable keypresses found')
        return BAD_FILE, keylog_clear

    def _is_next_key_equals_letter(self,
                                   firstkeystroke_index: pd.Index,
                                   i: int):
        next_key = self._experiment.key_logs.loc[firstkeystroke_index + ExperimentReader.TRIAL_LENGTH - 1 + i][
            'Letter']
        logging.debug(f'_is_next_key_equals_letter(firstkeystroke_index={firstkeystroke_index}, i={i})')
        return next_key == self._key.value

    def read(self, file_path: Path) -> RHDFile:
        rhd_file = RHDFile(fp=file_path,
                           start_time=self._rhd_file_start_time[file_path],
                           key_timings_adjusted=self._key_timings_adjusted[file_path],
                           absolute_timings=self._trial_timings_absolute[file_path])
        rhd_file.read()
        if not rhd_file.file_ok:
            self._append_error_message(rhd_file.file_path,
                                       ExperimentReader.ErrorMessageLevel.ERROR,
                                       'Incomplete RHD file')
        return rhd_file

