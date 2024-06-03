import logging
from pathlib import Path
import numpy as np
from common.folder_paths import VALID_WINDOWS_ROOT
from experiments_read.rhd_data import RHDData


class RHDWindowSlicer:
    def __init__(self, rhd_data: RHDData, win_length_seconds=0.2):
        self._rhd = rhd_data
        self._win_length_seconds: float = win_length_seconds
        self._windows: dict[str: np.ndarray] = {}

    @property
    def windows(self) -> dict[str, np.ndarray]:
        return self._windows

    @staticmethod
    def _save_window(fpath: Path, window: np.ndarray):

        logging.info(f'saving {fpath.as_posix()}')
        with open(fpath.as_posix(), "wb") as f:
            np.save(f, window)

    def slice(self):
        key_inds = np.argwhere(self._rhd.absolute_timings[:, -1] != "Key.space")
        for i, t in enumerate(self._rhd.key_timings_adjusted[key_inds]):
            win = self._slice_around_time(time=t[0])
            label = self._rhd.absolute_timings[key_inds[i], -1][0]
            self._windows[f'{self._rhd.name}_label_{label}_time_{t}'] = win

    def save_windows(self, root_path: Path = VALID_WINDOWS_ROOT):
        for k, w in self._windows.items():
            fp = root_path / f'{k}.npy'
            RHDWindowSlicer._save_window(fpath=fp, window=w)

    def _slice_around_time(self, time: float):
        if np.isin(round(float(time) - self._win_length_seconds / 2, 4), self._rhd.data_timestamps):
            if np.isin(round(float(time) + self._win_length_seconds / 2, 4), self._rhd.data_timestamps):
                # intermediate index
                start = np.where(self._rhd.data_timestamps == round(float(time) - self._win_length_seconds / 2, 4))[0][
                    0]
                end = np.where(self._rhd.data_timestamps == round(float(time) + self._win_length_seconds / 2, 4))[0][0]
            else:
                # last index
                end = len(self._rhd.data_timestamps)
                start = int(end - self._win_length_seconds * self._rhd.sampling_rate)
        else:
            # first index
            start = 0
            end = int(self._win_length_seconds * self._rhd.sampling_rate)

        logging.debug(f"Slice around time: {time} start: {start} end: {end}")
        return self._rhd.amplifier_data[:, start:end]
