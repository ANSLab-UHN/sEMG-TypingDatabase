import datetime
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
from keypressemg.common.types_defined import KeyPress


def read_file_to_list(file_path):
    """
    Reads a csv file and returns a list of lists.
    (taken as is from Classify.py)
    :param file_path: path to the csv file
    :return: list of lists
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip().startswith('2022')]
        data = []
        # Pull time portion from the total keylog line
        for item in lines:
            parts = item.split(' - ')
            time_part = parts[0].split()[1]
            letter = parts[1].strip("'")
            data.append([time_part, letter])
        df = pd.DataFrame(data, columns=['Time', 'Letter'])
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S,%f')
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []


def read_data(filename, measure_time=False):
    """Reads Intan Technologies RHD2000 data file generated by acquisition
    software (IntanRHX, or legacy Recording Controller / USB Evaluation
    board software).

    Data are returned in a dictionary, for future extensibility.
    (taken almost as is from load_intan_rhd_format.py)
    """
    import time
    from keypressemg.data_prep.intanutil.header import (read_header,
                                            header_to_result)
    from keypressemg.data_prep.intanutil.data import (calculate_data_size,
                                          read_all_data_blocks,
                                          check_end_of_file,
                                          parse_data,
                                          data_to_result)
    from keypressemg.data_prep.intanutil.filter import apply_notch_filter
    tic: Optional[float] = None
    if measure_time:
        # Start measuring how long this read takes.
        tic = time.time()

    # Open file for reading.
    with open(filename, 'rb') as fid:

        # Read header and summarize its contents to console.
        header = read_header(fid)

        # Calculate how much data is present and summarize to console.
        data_present, filesize, num_blocks, num_samples = (
            calculate_data_size(header, filename, fid))

        # If .rhd file contains data, read all present data blocks into 'data'
        # dict, and verify the amount of data read.
        if data_present:
            data = read_all_data_blocks(header, num_samples, num_blocks, fid)
            check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhd file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhd file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    if measure_time:
        toc = time.time()
        assert tic is not None, 'tic is not set'
        # Report how long read took.
        print('Done!  Elapsed time: {0:0.1f} seconds'.format(toc - tic))

    # Return 'result' dict.
    return result


def get_rhd_file_start_time(filename: str) -> datetime.datetime:
    return pd.to_datetime(f'{filename[9:11]}:{filename[11:13]}:{filename[13:15]}', format='%H:%M:%S')


def get_first_dataframe_row_index_after_base_time(base_time: datetime.datetime,
                                                  df: pd.DataFrame,
                                                  sorted_df_given: bool = False) -> pd.Index:
    assert 'Time' in df.columns, f'Expected dataframe to have Time column, got {df.columns}'

    df_sorted_by_time = df.sort_values(by='Time') if not sorted_df_given else df
    first_timestamp_after_base_time = df_sorted_by_time[df_sorted_by_time['Time'] >= base_time]['Time'].iloc[0]
    first_index_after_timestamp = df[df['Time'] == first_timestamp_after_base_time].index[0]
    logging.debug(f'get_first_dataframe_row_index_after_base_time base time {base_time} first_timestamp_after_base_time {first_timestamp_after_base_time} first_index_after_timestamp {first_index_after_timestamp} ')
    return first_index_after_timestamp


def is_first_key_after_index_not_too_long(df: pd.DataFrame, key: KeyPress,
                                          pd_index: pd.Index,
                                          too_long_time: datetime.timedelta) -> bool:
    # Look for
    # the next `key` press after
    # the time df row indexed by  pd_index (base_time = df.loc[pd_index]['Time']).
    # verify that it is not too long after that time
    too_long, found = False, False
    base_time = df.loc[pd_index]['Time']
    while not found and not too_long:
        row = df.iloc[pd_index]
        v, t = row['Letter'], row['Time']
        found = v == key.value
        pd_index += 1
        too_long = t > base_time + too_long_time
        logging.debug(f'is_first_key_after_index_not_too_long v {v} key {key.value} pd_index {pd_index} too long {too_long}')
    return not too_long

    # if too_long:
    #     err_df = pd.concat([err_df, pd.DataFrame({'Participant': [p],
    #                                               'Test': [t],
    #                                               'Filename': [f],
    #                                               'Error': 'Could not find start of trial'})])


def unique_letters_in_window(window_start_index: pd.Index,
                             window_length: int,
                             keylog: pd.DataFrame) \
        -> tuple[pd.DataFrame, np.ndarray[Any, np.dtype[int]], np.ndarray[Any, np.dtype[int]]]:
    assert 'Letter' in keylog.columns, f'Expected keylog contain a Letter Column'
    trial_window = keylog.loc[window_start_index:window_start_index + window_length]
    window_letters = np.array(trial_window['Letter'])
    unique_items, counts = np.unique(window_letters, return_counts=True)
    return trial_window, unique_items, counts


