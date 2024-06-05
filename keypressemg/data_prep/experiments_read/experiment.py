from pathlib import Path
import pandas as pd
from keypressemg.data_prep.experiments_read.utils import read_file_to_list
from keypressemg.common.types_defined import (Participant,
                                              DayT1T2,
                                              KeyPress)


class Experiment:
    """
    Represents a single experiment made by a participant e.g. P12/T1

    Note that each participant should have a pair of tests e.g. P12/T1 and P12/T2

    Each experiment contains a key log file, a lag timings csv file and
     a Data folder containing all rhd files recorded during this test

    arguments:
    folder_path (Path): Path to the folder containing the test data.
    participant (Participant): Participant to be tested.
    test (TestOneOrTwo): Test to be tested.
    """

    def __init__(self, folder_path: Path, participant: Participant, test: DayT1T2):
        self._folder_path = folder_path
        self._participant = participant
        self._test = test

        assert self.folder_path.exists(), f"{self.folder_path} does not exist"
        assert self.folder_path.is_dir(), f"{self.folder_path} is not a directory"

        # construct experiment folder path e.g.  home/user/experiments/P1/T2
        self._test_folder_path = folder_path / participant.value / test.value

        assert self._test_folder_path.exists(), f"{self._test_folder_path} does not exist"
        assert self._test_folder_path.is_dir(), f"{self._test_folder_path} is not a directory"

        self._data_path = self._test_folder_path / "Data"

        assert self._test_folder_path.exists(), f'{self._test_folder_path} does not exist'
        assert self._test_folder_path.is_dir(), f'{self._test_folder_path} is not a directory'

        self._key_logs: pd.DataFrame = read_file_to_list(self._test_folder_path / 'Keylogs.txt')
        self._lag_timings: pd.DataFrame = pd.read_csv(self._test_folder_path / 'LAG_TIMINGS.csv')
        self._key_logs_sorted: pd.DataFrame = self._key_logs.sort_values(by='Time')

    def get_letter_filepaths(self, key: KeyPress) -> list[Path]:
        return [f for f in self._data_path.glob('*.rhd') if f.stem.startswith(key.value.upper())]

    @property
    def test_folder_path(self) -> Path:
        return self._test_folder_path

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def test(self) -> DayT1T2:
        return self._test

    @property
    def participant(self) -> Participant:
        return self._participant

    @property
    def folder_path(self) -> Path:
        return self._folder_path

    @property
    def key_logs(self) -> pd.DataFrame:
        return self._key_logs

    @property
    def key_logs_sorted(self) -> pd.DataFrame:
        return self._key_logs_sorted

    @property
    def lag_timings(self) -> pd.DataFrame:
        return self._lag_timings
