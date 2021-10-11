'''NOTE: This module is deprecated, use csfr module instead.'''

import h5py
import numpy as np
import os
import sys

from spta.util import log as log_util
from spta.util import maths as maths_util

# TODO extract this info into a class that can interface with spt_metadata

# Raw dataset and its properties
RAW_DATASET = 'raw/TEMPERATURE_1979-2015.hdf'
DATASET_NAME = 'SouthAmerica'
DATASET_YEAR_START = 1979
DATASET_YEAR_END = 2015
DATASET_SAMPLES_PER_DAY = 4

# some metadata
FORMAT_BY_YEAR = 'raw/{}_{}_{}_{}spd.npy'


def read_hdf5_dataset(series_start, series_end, dataset_source):
    '''
    Reads from hdf5 file, entire coordinates, using the sample interval series_start:series_end.
    '''
    logger = log_util.logger_for_me(read_hdf5_dataset)

    dataset_str = '{}[{}:{}]'.format(dataset_source, series_start, series_end)
    logger.debug('Reading dataset: {}'.format(dataset_str))

    if not os.path.exists(dataset_source):
        msg = 'Dataset not found %s' % dataset_source
        raise ValueError(msg)

    with h5py.File(dataset_source, 'r') as f:
        real = f['real'][...]
        logger.info('field "real" of dataset has shape: {}'.format(real.shape))
        real = real[series_start:series_end, :, :]

    series_len = series_end - series_start
    logger.info('Read dataset {} with {} samples: {}'.format(dataset_str, series_len, real.shape))
    return real


def read_hdf5_dataset_by_year(year_start, year_end, dataset_source=RAW_DATASET):
    '''
    Reads from hdf5 file, entire coordinates, using the start and end year.
    '''
    # check boundaries and sanity
    if year_start > year_end or year_start < DATASET_YEAR_START or year_end > DATASET_YEAR_END:
        raise ValueError('Invalid year interval: {}-{}'.format(year_start, year_end))

    # this transforms years to samples
    (series_start, series_end) = \
        maths_util.years_to_series_interval(year_start=year_start,
                                            year_end=year_end,
                                            first_year_in_sample=DATASET_YEAR_START,
                                            samples_per_day=DATASET_SAMPLES_PER_DAY)

    return read_hdf5_dataset(series_start, series_end, dataset_source)


def save_dataset_interval(dataset_interval, name, year_start, year_end, spd):
    '''
    Stores a previously loaded dataset interval as a numpy file.

    dataset_interval
        a subset of the entire dataset, by time interval

    year_start
        first year of interval

    year_end
        last year of interval (inclusive)

    spd
        current samples per day
    '''
    dataset_filename = FORMAT_BY_YEAR.format(name, year_start, year_end, spd)
    np.save(dataset_filename, dataset_interval)

    logger = log_util.logger_for_me(save_dataset_interval)
    logger.info('Saved dataset: {}'.format(dataset_filename))

    return dataset_filename


def try_load_dataset_interval(name, year_start, year_end, spd):
    '''
    Tries to loads a previously saved dataset interval from its numpy file.
    If the data was not saved before, returns None without error.
    '''
    logger = log_util.logger_for_me(try_load_dataset_interval)
    dataset_filename = FORMAT_BY_YEAR.format(name, year_start, year_end, spd)

    dataset_interval = None

    try:
        if os.path.exists(dataset_filename):
            dataset_interval = np.load(dataset_filename)
        else:
            logger.debug('Saved dataset not found: {}'.dataset_filename)

    except Exception:
        logger.warn('Error loading dataset: {}'.format(dataset_filename))

    return dataset_interval, dataset_filename


def retrieve_dataset_interval(year_start, year_end, spd, name=DATASET_NAME):
    '''
    Retrieves a subset of the dataset, given the year interval. Tries to load from a previously
    saved file, if this fails then the data is first extracted from the entire dataset and saved
    for future use.

    Assumes spd = 1 or spd = 4!
    '''
    logger = log_util.logger_for_me(retrieve_dataset_interval)

    # big assumption
    assert spd == 1 or spd == 4

    dataset_interval, dataset_filename = try_load_dataset_interval(name, year_start, year_end, spd)
    if dataset_interval is None:

        # the attempt to load a previously saved dataset failed, extract data from whole dataset
        dataset_interval_4spd = read_hdf5_dataset_by_year(year_start=year_start,
                                                          year_end=year_end,
                                                          dataset_source=RAW_DATASET)

        # convert to 1spd?
        if spd == 1:
            dataset_interval = average_4spd_to_1spd(dataset_interval_4spd)
        else:
            dataset_interval = dataset_interval_4spd

        # save this dataset interval to avoid reading the whole dataset again
        dataset_filename = save_dataset_interval(dataset_interval, name, year_start, year_end, spd)

    logger.info('Loaded dataset: {} -> {}'.format(dataset_filename, dataset_interval.shape))
    return dataset_interval


def average_4spd_to_1spd(numpy_dataset):
    '''
    Given a spatio temporal region with the defaults of 4 points per day (spd=4), average the
    points in each day to get 1 point per day (spd = 1)
    '''
    logger = log_util.logger_for_me(average_4spd_to_1spd)

    (series_len, x_len, y_len) = numpy_dataset.shape

    # we have 4 points per day
    # average these four points to get a smoother curve
    new_series_len = int(series_len / 4)
    single_point_per_day = np.empty((new_series_len, x_len, y_len))

    for x in range(0, x_len):
        for y in range(0, y_len):
            point_series = numpy_dataset[:, x, y]
            series_reshape = (new_series_len, 4)
            smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
            # sptr.log.debug('smooth: %s' % smooth)
            single_point_per_day[:, x, y] = np.array(smooth)

    log_msg = 'Reshaped {} (4spd) -> {} (1spd)'
    logger.info(log_msg.format(numpy_dataset.shape, single_point_per_day.shape))

    return single_point_per_day


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: {} <year_start> <year_end> ["spd1"]'.format(sys.argv[0]))
        sys.exit(1)

    import logging
    import time

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    t_start = time.time()

    year_start = int(sys.argv[1])
    year_end = int(sys.argv[2])

    spd = 4
    if len(sys.argv) > 3:
        spd = 1

    dataset_interval = retrieve_dataset_interval(year_start, year_end, spd)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: {} seconds'.format(elapsed))
