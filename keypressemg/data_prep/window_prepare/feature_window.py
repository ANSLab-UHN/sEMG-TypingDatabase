import logging
from pathlib import Path
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def feature_extract(data: np.ndarray, features: list[str]) -> dict[str, np.ndarray]:
    """
    taken almost as is from featextract in Classify.py
    """

    feat = {}
    threshWAMP = 6
    threshZC = 10
    threshSSC = 8
    # RMS
    if "RMS" in features:
        rms = np.sqrt(np.mean(np.power(data, 2)))
        feat["RMS"] = rms
    if "MAV" in features:
        mav = np.mean(abs(data))
        feat["MAV"] = mav
    # WL
    if "WL" in features:
        feat["WL"] = np.sum(np.abs(data[1:] - data[:-1]))
    # LOGVAR
    if "LOGVAR" in features:
        LogVar = np.log(np.var(data))
        feat["LOGVAR"] = LogVar
    #AR
    if "AR1" in features:
        ar_model = AutoReg(data, 1).fit()
        AR = ar_model.params
        feat["AR1"] = AR[1]
    if "AR2" in features:
        ar_model = AutoReg(data, 2).fit()
        AR = ar_model.params
        feat["AR2"] = AR[2]
    if "AR3" in features:
        ar_model = AutoReg(data, 3).fit()
        AR = ar_model.params
        feat["AR3"] = AR[3]
    if "AR4" in features:
        ar_model = AutoReg(data, 4).fit()
        AR = ar_model.params
        feat["AR4"] = AR[4]
    #WAMP
    if "WAMP" in features:
        feat["WAMP"] = np.sum(np.abs(data[1:] - data[:-1]) > threshWAMP)
    #SSC
    if "SSC" in features:
        SSC = 0
        for i in range(1, len(data) - 1):
            if (np.sign(data[i] - data[i - 1]) * np.sign(data[i] - data[i + 1]) == 1) and (
                    abs(data[i] - data[i - 1]) > threshSSC or abs(data[i] - data[i + 1]) > threshSSC):
                SSC += 1
        feat["SSC"] = SSC
    #ZC
    if "ZC" in features:
        ZC = 0
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        for i in zero_crossings:
            if np.abs(data[i] - data[i + 1]) > threshZC:
                ZC += 1
        feat["ZC"] = ZC
    #VAR
    if "VAR" in features:
        var = np.sum(np.power(data, 2)) / (len(data) - 1)
        feat["VAR"] = var
    #IEMG
    if "IEMG" in features:
        iemg = np.sum(np.abs(data))
        feat["IEMG"] = iemg
    #MAVS
    if "MAVS" in features:
        k = 3
        curindx = 0
        segment_length = np.floor(len(data) / k)
        mavs = np.zeros(k - 1)
        for i in mavs:
            i = np.mean(abs(data[int(curindx + segment_length):int(curindx + 2 * segment_length)])) - np.mean(
                abs(data[int(curindx):int(curindx + segment_length)]))
            curindx += segment_length
        feat["MAVS"] = np.mean(mavs)
    return feat


class FeatureWindow:
    # FEATURES = ['RMS', 'LOGVAR', 'WL', 'WAMP']
    FEATURES = ['RMS', 'LOGVAR', 'WL', 'WAMP', 'AR1', 'AR2']

    def __init__(self, window_path: Path, feature_windows_path: Path):
        assert window_path.exists(), f"{window_path} does not exist"
        assert window_path.is_file(), f"{window_path} is not a file"
        assert 'npy' in window_path.suffix, f"{window_path} is not a npy file"
        self.window_path = window_path

        if not feature_windows_path.exists():
            feature_windows_path.mkdir()
        assert feature_windows_path.exists(), f"{feature_windows_path} does not exist"
        assert feature_windows_path.is_dir(), f"{feature_windows_path} is not a directory"
        self.feature_windows_path = feature_windows_path

        self.window_data = None
        self.window_features = None

    def load_window(self):
        with open(self.window_path, 'rb') as f:
            self.window_data = np.load(f)

    def extract_window_features(self):
        assert self.window_data is not None, f'window not loaded yet'

        self.window_features = np.array([feature_extract(self.window_data[i, :], FeatureWindow.FEATURES)[j]
                                         for i in range(self.window_data.shape[0])
                                         for j in FeatureWindow.FEATURES])

    def save(self):
        assert self.window_features is not None, 'features not extracted yet'
        fp = self.feature_windows_path / f'{self.window_path.stem}.npy'
        logging.info(f'saving {fp.as_posix()}')
        with open(fp.as_posix(), "wb") as f:
            np.save(f, self.window_features)
