import logging
from pathlib import Path
import numpy as np
from keypressemg.common.folder_paths import VALID_EXPERIMENTS_ROOT
from keypressemg.data_prep.experiments_read.rhd_data import RHDData
from keypressemg.data_prep.experiments_read.utils import read_data


class RHDFile:
    def __init__(self, fp: Path, start_time=None, key_timings_adjusted=None, absolute_timings=None):
        assert fp.exists(), f'File {fp} does not exist'
        assert fp.is_file(), f'File {fp} is not a file'
        self._file_path = fp
        self._save_path = VALID_EXPERIMENTS_ROOT / f'{fp.parent.parent.parent.stem}_{fp.parent.parent.stem}_{fp.stem}.npy'
        self._amplifier_data: np.ndarray = np.array([])
        self._data_timestamps: np.ndarray = np.array([])
        self._sampling_rate: float = 0.0
        self._file_ok: bool = False
        self._start_time = np.array([start_time])
        self._key_timings_adjusted = key_timings_adjusted
        self._absolute_timings = absolute_timings

        self._rhd_data: RHDData = RHDData()

    def read(self):

        try:
            datafile = read_data(self._file_path.as_posix())
            self._rhd_data.name = self._save_path.stem
            self._rhd_data.sampling_rate = datafile['frequency_parameters']['amplifier_sample_rate']
            self._rhd_data.data_timestamps = datafile["t_amplifier"]
            self._rhd_data.amplifier_data = datafile["amplifier_data"]
            self._rhd_data.absolute_timings = self._absolute_timings
            self._rhd_data.key_timings_adjusted = self._key_timings_adjusted
            self._rhd_data.start_time = self._start_time
            self._file_ok = True
        except Exception as e:
            logging.exception(f"Exception {e}\nError reading {self.file_path.as_posix()}")
            self._file_ok = False

    @property
    def rhd_data(self):
        return self._rhd_data

    @property
    def file_path(self):
        return self._file_path

    @property
    def save_path(self):
        return self._save_path

    @property
    def file_ok(self):
        return self._file_ok

    @property
    def amplifier_data(self):
        return self._rhd_data.amplifier_data

    @property
    def data_timestamps(self):
        return self._rhd_data.data_timestamps

    @property
    def sampling_rate(self):
        return self._rhd_data.sampling_rate

    @property
    def absolute_timings(self):
        return self._rhd_data.absolute_timings

    @property
    def key_timings_adjusted(self):
        return self._rhd_data.key_timings_adjusted


