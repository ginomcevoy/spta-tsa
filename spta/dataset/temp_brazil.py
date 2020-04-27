import h5py
import numpy as np
import os
import time
import sys


DATASET = 'raw/TEMPERATURE_1979-2015.hdf'
# REDUCED_FORMAT = 'raw/dataset_%s.npy'
REDUCED_FORMAT = 'raw/brazil_{}y_{}ppd.npy'
REDUCED_FORMAT_LAST = 'raw/brazil_{}y_{}ppd_last.npy'
POINTS_PER_DAY = 4
POINTS_PER_YEAR = 365 * POINTS_PER_DAY


def load_dataset(filename, series_len):
    '''
    Reads from hdf file, entire region, series_len number of points
    '''
    if not os.path.exists(filename):
        msg = 'Dataset not found %s' % filename
        raise ValueError(msg)

    with h5py.File(filename, 'r') as f:
        real = f['real'][...]
        real = real[:series_len, :, :]

    print('Shape of %s with %d points: %s' % (filename, series_len, real.shape))
    return real


def load_dataset_last(filename, series_len):
    '''
    Reads from hdf file, entire region, series_len number of points
    '''
    if not os.path.exists(filename):
        msg = 'Dataset not found %s' % filename
        raise ValueError(msg)

    with h5py.File(filename, 'r') as f:
        real = f['real'][...]
        real = real[-series_len:, :, :]

    print('Shape of %s with %d points: %s' % (filename, series_len, real.shape))
    return real


def save(filename, data):
    np.save(filename, data)


def save_reduced_dataset(output, series_len):
    print('Reading dataset: {}'.format(DATASET))
    data = load_dataset(DATASET, series_len)

    print('Saving reduced dataset to: {}'.format(output))
    np.save(output, data)


def save_reduced_dataset_last(output, series_len):
    print('Reading dataset (last years): {}'.format(DATASET))
    data = load_dataset_last(DATASET, series_len)

    print('Saving reduced dataset to: {}'.format(output))
    np.save(output, data)


def load_saved(filename):
    if not os.path.exists(filename):
        msg = 'Dataset not found %s' % filename
        raise ValueError(msg)

    data = np.load(filename)
    print('Shape of %s with: %s' % (filename, data.shape))
    return data


def load_brazil_temps(years, ppd=4):
    filename = REDUCED_FORMAT.format(years, ppd)
    return load_saved(filename)


def load_brazil_temps_last(years, ppd=4):
    filename = REDUCED_FORMAT_LAST.format(years, ppd)
    return load_saved(filename)


def load_with_len(series_len):
    filename = REDUCED_FORMAT % str(series_len)
    return load_saved(filename)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: {} <years> ["last"]'.format(sys.argv[0]))
        sys.exit(1)

    t_start = time.time()

    years = int(sys.argv[1])
    series_len = POINTS_PER_YEAR * years

    # fixed for now
    ppd = 4

    if len(sys.argv) == 3 and sys.argv[2] == 'last':
        filename = REDUCED_FORMAT_LAST.format(years, ppd)
        save_reduced_dataset_last(filename, series_len)
    else:
        filename = REDUCED_FORMAT.format(years, ppd)
        save_reduced_dataset(filename, series_len)

    # # 4 years of data
    # pts_4y = POINTS_PER_YEAR * 4
    # filename_4y = REDUCED_FORMAT % pts_4y
    # save_reduced_dataset(filename_4y, pts_4y)

    # # 1 year of data
    # pts_1y = POINTS_PER_YEAR
    # filename_1y = REDUCED_FORMAT % pts_1y
    # save_reduced_dataset(filename_1y, pts_1y)

    # load_saved('dataset_4000.npy')
    # load_with_len(4000)

    # # test opening the file and reading the shape
    # brazil_data = load_brazil_temps(years, ppd)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s seconds' % str(elapsed))
