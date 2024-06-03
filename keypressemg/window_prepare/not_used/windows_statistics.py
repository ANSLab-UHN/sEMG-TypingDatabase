import logging
from pathlib import Path
import numpy as np

from common.folder_paths import VALID_FEATURES_ROOT


class WindowsFolderStatistics:
    NUM_CHANNELS: int = 96

    def __init__(self, windows_folder: Path) -> None:
        assert windows_folder.exists(), f'{windows_folder.as_posix()} does not exist'
        assert windows_folder.is_dir(), f'{windows_folder.as_posix()} is not a directory'

        self._windows_folder = windows_folder
        self._windows_file_gen = self._windows_folder.glob('*].npy')

    def calculate_avg(self):
        window_vector: np.ndarray
        channel_sum = np.zeros(shape=(self.NUM_CHANNELS,), dtype=float)
        num_files = 0
        windows_file_gen = self._windows_folder.glob('*].npy')
        # calculate average per channel or feature
        for window_file in windows_file_gen:
            with open(window_file.as_posix(), 'rb') as f:
                window_vector = np.load(f)
            logging.info(f'window vector shape: {window_vector.shape}')
            # logging.info(f'window vector : {window_vector}')
            channel_sum += window_vector
            # logging.info(f'channel sum : {channel_sum}')
            num_files += 1
        channel_avg = channel_sum / float(num_files)
        logging.info(f'avg channel shape : {channel_avg.shape}')
        # logging.info(f'average window vector: {channel_avg}')
        with open((self._windows_folder / 'channel_avg.npy').as_posix(), 'wb') as f:
            np.save(f, channel_avg)

    def calculate_std(self):
        with open((self._windows_folder / 'channel_avg.npy').as_posix(), 'rb') as f:
            channel_avg = np.load(f)
        logging.info(f'average window vector shape: {channel_avg.shape}')
        num_files = 0
        windows_file_gen = self._windows_folder.glob('*].npy')
        # calculate std per channel or feature
        channel_sq_diff_sum = np.zeros(shape=(self.NUM_CHANNELS, ), dtype=float)
        for window_file in windows_file_gen:
            with open(window_file.as_posix(), 'rb') as f:
                window_vector = np.load(f)
            num_files += 1
            channel_sq_diff_sum += np.square(np.subtract(window_vector, channel_avg))
            # logging.info(f' channel_sq_diff_sum: {channel_sq_diff_sum}')
        channel_std = np.sqrt(channel_sq_diff_sum / float(num_files))
        logging.info(f'std window vector: {channel_std}')
        with open((self._windows_folder / 'channel_std.npy').as_posix(), 'wb') as f:
            np.save(f, channel_std)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    wfs = WindowsFolderStatistics(VALID_FEATURES_ROOT)
    wfs.calculate_avg()
    wfs.calculate_std()
