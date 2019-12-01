import h5py
import numpy as np
import os
import time


DATASET = 'raw/TEMPERATURE_1979-2015.hdf'
REDUCED_FORMAT = 'raw/dataset_%s.npy'
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


def save(filename, data):
    np.save(filename, data)


def save_reduced_dataset(output, series_len):
    data = load_dataset(DATASET, series_len)
    np.save(output, data)


def load_saved(filename):
    if not os.path.exists(filename):
        msg = 'Dataset not found %s' % filename
        raise ValueError(msg)

    data = np.load(filename)
    print('Shape of %s with: %s' % (filename, data.shape))
    return data


def load_with_len(series_len):
    filename = REDUCED_FORMAT % str(series_len)
    return load_saved(filename)


if __name__ == '__main__':
    t_start = time.time()

    # # 4 years of data
    # pts_4y = POINTS_PER_YEAR * 4
    # filename_4y = REDUCED_FORMAT % pts_4y
    # save_reduced_dataset(filename_4y, pts_4y)

    # 1 year of data
    pts_1y = POINTS_PER_YEAR
    filename_1y = REDUCED_FORMAT % pts_1y
    save_reduced_dataset(filename_1y, pts_1y)

    # load_saved('dataset_4000.npy')
    # load_with_len(4000)
    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s' % str(elapsed))
