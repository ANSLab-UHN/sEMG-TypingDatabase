import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class RHDData:
    name: str = ''
    amplifier_data = None
    data_timestamps = None
    sampling_rate: float = 0.0
    absolute_timings = None
    key_timings_adjusted = None
    start_time = None


class RHDDataFileLoader:
    def __init__(self, filepath: Path):
        self._file_path = filepath
        self._data = RHDData()

    def load(self):
        with open(self._file_path.as_posix(), 'rb') as f:
            self._data.name = self._file_path.stem
            self._data.amplifier_data = np.load(f)
            self._data.data_timestamps = np.load(f)
            self._data.sampling_rate = np.load(f)
            self._data.key_timings_adjusted = np.load(f)
            self._data.absolute_timings = np.load(f, allow_pickle=True)
            self._data.start_time = np.load(f, allow_pickle=True)
        logging.info(f'Loaded {self._data.name}')

    @property
    def data(self) -> RHDData:
        assert self._data is not None, f'Data not loaded'
        return self._data


class RHDDataFileSaver:
    def __init__(self, save_path: Path, rhd: RHDData, replace=True):
        assert save_path.parent.exists(), f'File {save_path.parent} does not exist'
        self._data = rhd
        assert save_path.exists(), f'Path {save_path.as_posix()} does not exist'
        assert save_path.is_dir(), f'Path {save_path.as_posix()} is expected to be a folder'
        self._file_path = save_path / f'{rhd.name}.npy'
        assert replace or not save_path.exists(), f'File {save_path.as_posix()} exists and replace is False'

    def save(self):
        logging.info(f'Saving {self._file_path.stem} to {self._file_path.parent}')
        with open(self._file_path.as_posix(), 'wb') as f:
            np.save(f, self._data.amplifier_data)
            np.save(f, self._data.data_timestamps)
            np.save(f, self._data.sampling_rate)
            np.save(f, self._data.key_timings_adjusted)
            np.save(f, self._data.absolute_timings)
            np.save(f, self._data.start_time)
        logging.info(f'{self._file_path.as_posix()} saved.')
