# Adrian Foy September 2023

"""Interacts with RHD data, both directly at the binary level with RHD data
blocks and at the Python level with dictionaries of NumPy arrays.
"""


import os
import struct

import numpy as np

from keypressemg.data_prep.intanutil.header import get_timestamp_signed
from keypressemg.data_prep.intanutil.report import print_progress


def calculate_data_size(header, filename, fid):
    """Calculates how much data is present in this file. Returns:
    data_present: Bool, whether any data is present in file
    filesize: Int, size (in bytes) of file
    num_blocks: Int, number of 60 or 128-sample data blocks present
    num_samples: Int, number of samples present in file
    """
    bytes_per_block = get_bytes_per_data_block(header)

    # Determine filesize and if any data is present.
    filesize = os.path.getsize(filename)
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    # If the file size is somehow different than expected, raise an error.
    if bytes_remaining % bytes_per_block != 0:
        raise FileSizeError(
            'Something is wrong with file size : '
            'should have a whole number of data blocks')

    # Calculate how many data blocks are present.
    num_blocks = int(bytes_remaining / bytes_per_block)

    num_samples = calculate_num_samples(header, num_blocks)

    # print_record_time_summary(num_samples['amplifier'],
    #                           header['sample_rate'],
    #                           data_present)

    return data_present, filesize, num_blocks, num_samples


def read_all_data_blocks(header, num_samples, num_blocks, fid):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.
    """
    data, indices = initialize_memory(header, num_samples)
    #print("Reading data from file...")
    print_step = 10
    percent_done = print_step
    for i in range(num_blocks):
        read_one_data_block(data, header, indices, fid)
        advance_indices(indices, header['num_samples_per_data_block'])
        percent_done = print_progress(i, num_blocks, print_step, percent_done)
    return data


def check_end_of_file(filesize, fid):
    """Checks that the end of the file was reached at the expected position.
    If not, raise FileSizeError.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)
    """
    #print('Parsing data...')
    extract_digital_data(header, data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)


def data_to_result(header, data, result):
    """Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.
    """
    if header['num_amplifier_channels'] > 0:
        result['t_amplifier'] = data['t_amplifier']
        result['amplifier_data'] = data['amplifier_data']

    if header['num_aux_input_channels'] > 0:
        result['t_aux_input'] = data['t_aux_input']
        result['aux_input_data'] = data['aux_input_data']

    if header['num_supply_voltage_channels'] > 0:
        result['t_supply_voltage'] = data['t_supply_voltage']
        result['supply_voltage_data'] = data['supply_voltage_data']

    if header['num_temp_sensor_channels'] > 0:
        result['t_temp_sensor'] = data['t_temp_sensor']

    if header['num_board_adc_channels'] > 0:
        result['t_board_adc'] = data['t_board_adc']
        result['board_adc_data'] = data['board_adc_data']

    if (header['num_board_dig_in_channels'] > 0
            or header['num_board_dig_out_channels'] > 0):
        result['t_dig'] = data['t_dig']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_data'] = data['board_dig_in_data']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_data'] = data['board_dig_out_data']

    return result


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60 or 128 sample datablock."""
    # Depending on the system used to acquire the data,
    # 'num_samples_per_data_block' will be either 60 (USB Interface Board)
    # or 128 (Recording Controller).
    # Use this number along with numbers of channels to accrue a sum of how
    # many bytes each data block should contain.

    # Timestamps (one channel always present): Start with 4 bytes per sample.
    bytes_per_block = bytes_per_signal_type(
        header['num_samples_per_data_block'],
        1,
        4)

    # Amplifier data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_amplifier_channels'],
        2)

    # Auxiliary data: Add 2 bytes per sample per enabled aux input channel.
    # Note that aux inputs are sample 4x slower than amplifiers, so there
    # are 1/4 as many samples.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'] / 4,
        header['num_aux_input_channels'],
        2)

    # Supply voltage: Add 2 bytes per sample per enabled vdd channel.
    # Note that aux inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    bytes_per_block += bytes_per_signal_type(
        1,
        header['num_supply_voltage_channels'],
        2)

    # Analog inputs: Add 2 bytes per sample per enabled analog input channel.
    bytes_per_block += bytes_per_signal_type(
        header['num_samples_per_data_block'],
        header['num_board_adc_channels'],
        2)

    # Digital inputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Digital outputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            header['num_samples_per_data_block'],
            1,
            2)

    # Temp sensor: Add 2 bytes per sample per enabled temp sensor channel.
    # Note that temp sensor inputs are sampled once per data block
    # (60x or 128x slower than amplifiers), so there are
    # 1/60 or 1/128 as many samples.
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            1,
            header['num_temp_sensor_channels'],
            2)

    return bytes_per_block


def bytes_per_signal_type(num_samples, num_channels, bytes_per_sample):
    """Calculates the number of bytes, per data block, for a signal type
    provided the number of samples (per data block), the number of enabled
    channels, and the size of each sample in bytes.
    """
    return num_samples * num_channels * bytes_per_sample


def read_one_data_block(data, header, indices, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by indices."""
    samples_per_block = header['num_samples_per_data_block']

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data
    read_timestamps(fid,
                    data,
                    indices,
                    samples_per_block,
                    get_timestamp_signed(header))

    read_analog_signals(fid,
                        data,
                        indices,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         indices,
                         samples_per_block,
                         header)


def read_timestamps(fid, data, indices, num_samples, timestamp_signed):
    """Reads timestamps from binary file as a NumPy array, indexing them
    into 'data'.
    """
    start = indices['amplifier']
    end = start + num_samples
    format_sign = 'i' if timestamp_signed else 'I'
    format_expression = '<' + format_sign * num_samples
    read_length = 4 * num_samples
    data['t_amplifier'][start:end] = np.array(struct.unpack(
        format_expression, fid.read(read_length)))


def read_analog_signals(fid, data, indices, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            indices['amplifier'],
                            samples_per_block,
                            header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['aux_input_data'],
                            indices['aux_input'],
                            int(samples_per_block / 4),
                            header['num_aux_input_channels'])

    read_analog_signal_type(fid,
                            data['supply_voltage_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_supply_voltage_channels'])

    read_analog_signal_type(fid,
                            data['temp_sensor_data'],
                            indices['supply_voltage'],
                            1,
                            header['num_temp_sensor_channels'])

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            indices['board_adc'],
                            samples_per_block,
                            header['num_board_adc_channels'])


def read_digital_signals(fid, data, indices, samples_per_block, header):
    """Reads all digital signal types present in RHD files: board_dig_in_raw
    and board_dig_out_raw, into 'data' dict.
    """

    read_digital_signal_type(fid,
                             data['board_dig_in_raw'],
                             indices['board_dig_in'],
                             samples_per_block,
                             header['num_board_dig_in_channels'])

    read_digital_signal_type(fid,
                             data['board_dig_out_raw'],
                             indices['board_dig_out'],
                             samples_per_block,
                             header['num_board_dig_out_channels'])


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples*num_channels)
    dest[range(num_channels), start:end] = (
        tmp.reshape(num_channels, num_samples))


def read_digital_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be a digital signal type within 'data', either
    data['board_dig_in_raw'] or data['board_dig_out_raw'].
    """

    if num_channels < 1:
        return
    end = start + num_samples
    dest[start:end] = np.array(struct.unpack(
        '<' + 'H' * num_samples, fid.read(2 * num_samples)))


def calculate_num_samples(header, num_data_blocks):
    """Calculates number of samples for each signal type, storing the results
    in num_samples dict for later use.
    """
    samples_per_block = header['num_samples_per_data_block']
    num_samples = {}
    num_samples['amplifier'] = int(samples_per_block * num_data_blocks)
    num_samples['aux_input'] = int((samples_per_block / 4) * num_data_blocks)
    num_samples['supply_voltage'] = int(num_data_blocks)
    num_samples['board_adc'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_in'] = int(samples_per_block * num_data_blocks)
    num_samples['board_dig_out'] = int(samples_per_block * num_data_blocks)
    return num_samples


def initialize_memory(header, num_samples):
    """Pre-allocates NumPy arrays for each signal type that will be filled
    during this read, and initializes unique indices for data access to each
    signal type.
    """
    #print('\nAllocating memory for data...')
    data = {}

    # Create zero array for amplifier timestamps.
    t_dtype = np.int_ if get_timestamp_signed(header) else np.uint
    data['t_amplifier'] = np.zeros(num_samples['amplifier'], t_dtype)

    # Create zero array for amplifier data.
    data['amplifier_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples['amplifier']],
        dtype=np.uint)

    # Create zero array for aux input data.
    data['aux_input_data'] = np.zeros(
        [header['num_aux_input_channels'], num_samples['aux_input']],
        dtype=np.uint)

    # Create zero array for supply voltage data.
    data['supply_voltage_data'] = np.zeros(
        [header['num_supply_voltage_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for temp sensor data.
    data['temp_sensor_data'] = np.zeros(
        [header['num_temp_sensor_channels'], num_samples['supply_voltage']],
        dtype=np.uint)

    # Create zero array for board ADC data.
    data['board_adc_data'] = np.zeros(
        [header['num_board_adc_channels'], num_samples['board_adc']],
        dtype=np.uint)

    # By default, this script interprets digital events (digital inputs
    # and outputs) as booleans. if unsigned int values are preferred
    # (0 for False, 1 for True), replace the 'dtype=np.bool_' argument
    # with 'dtype=np.uint' as shown.
    # The commented lines below illustrate this for digital input data;
    # the same can be done for digital out.

    # data['board_dig_in_data'] = np.zeros(
    #     [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
    #     dtype=np.uint)
    # Create 16-row zero array for digital in data, and 1-row zero array for
    # raw digital in data (each bit of 16-bit entry represents a different
    # digital input.)
    data['board_dig_in_data'] = np.zeros(
        [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
        dtype=np.bool_)
    data['board_dig_in_raw'] = np.zeros(
        num_samples['board_dig_in'],
        dtype=np.uint)

    # Create 16-row zero array for digital out data, and 1-row zero array for
    # raw digital out data (each bit of 16-bit entry represents a different
    # digital output.)
    data['board_dig_out_data'] = np.zeros(
        [header['num_board_dig_out_channels'], num_samples['board_dig_out']],
        dtype=np.bool_)
    data['board_dig_out_raw'] = np.zeros(
        num_samples['board_dig_out'],
        dtype=np.uint)

    # Create dict containing each signal type's indices, and set all to zero.
    indices = {}
    indices['amplifier'] = 0
    indices['aux_input'] = 0
    indices['supply_voltage'] = 0
    indices['board_adc'] = 0
    indices['board_dig_in'] = 0
    indices['board_dig_out'] = 0

    return data, indices


def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
    if num_gaps == 0:
        print('No missing timestamps in data.')
    else:
        print('Warning: {0} gaps in timestamp data found.  '
              'Time scale will not be uniform!'
              .format(num_gaps))

    # Scale time steps (units = seconds).
    data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
    data['t_aux_input'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), 4)]
    data['t_supply_voltage'] = data['t_amplifier'][range(
        0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
    data['t_board_adc'] = data['t_amplifier']
    data['t_dig'] = data['t_amplifier']
    data['t_temp_sensor'] = data['t_supply_voltage']


def scale_analog_data(header, data):
    """Scales all analog data signal types (amplifier data, aux input data,
    supply voltage data, board ADC data, and temp sensor data) to suitable
    units (microVolts, Volts, deg C).
    """
    # Scale amplifier data (units = microVolts).
    data['amplifier_data'] = np.multiply(
        0.195, (data['amplifier_data'].astype(np.int32) - 32768))

    # Scale aux input data (units = Volts).
    data['aux_input_data'] = np.multiply(
        37.4e-6, data['aux_input_data'])

    # Scale supply voltage data (units = Volts).
    data['supply_voltage_data'] = np.multiply(
        74.8e-6, data['supply_voltage_data'])

    # Scale board ADC data (units = Volts).
    if header['eval_board_mode'] == 1:
        data['board_adc_data'] = np.multiply(
            152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    elif header['eval_board_mode'] == 13:
        data['board_adc_data'] = np.multiply(
            312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768))
    else:
        data['board_adc_data'] = np.multiply(
            50.354e-6, data['board_adc_data'])

    # Scale temp sensor data (units = deg C).
    data['temp_sensor_data'] = np.multiply(
        0.01, data['temp_sensor_data'])


def extract_digital_data(header, data):
    """Extracts digital data from raw (a single 16-bit vector where each bit
    represents a separate digital input channel) to a more user-friendly 16-row
    list where each row represents a separate digital input channel. Applies to
    digital input and digital output data.
    """
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_in_raw'],
                (1 << header['board_dig_in_channels'][i]['native_order'])
                ),
            0)

    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_out_raw'],
                (1 << header['board_dig_out_channels'][i]['native_order'])
                ),
            0)


def advance_indices(indices, samples_per_block):
    """Advances indices used for data access by suitable values per data block.
    """
    # Signal types sampled at the sample rate:
    # Index should be incremented by samples_per_block every data block.
    indices['amplifier'] += samples_per_block
    indices['board_adc'] += samples_per_block
    indices['board_dig_in'] += samples_per_block
    indices['board_dig_out'] += samples_per_block

    # Signal types sampled at 1/4 the sample rate:
    # Index should be incremented by samples_per_block / 4 every data block.
    indices['aux_input'] += int(samples_per_block / 4)

    # Signal types sampled once per data block:
    # Index should be incremented by 1 every data block.
    indices['supply_voltage'] += 1


class FileSizeError(Exception):
    """Exception returned when file reading fails due to the file size
    being invalid or the calculated file size differing from the actual
    file size.
    """
