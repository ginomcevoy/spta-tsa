import numpy as np
import time
from collections import namedtuple

import dataset as ds

Region = namedtuple('Region', 'x1, x2, y1, y2')


def get_region_data(dataset, region):
    return dataset[:, region.x1:region.x2, region.y1:region.y2]


def get_sao_paulo_data(dataset):
    sp_region = Region(55, 75, 50, 70)
    # print(sp_region)
    sp_data = get_region_data(dataset, sp_region)
    print('SP: %s' % (sp_data.shape,))
    return sp_data


def get_10p_3x3():
    ds4y = ds.load_with_len(5840)
    small_region = Region(55, 58, 50, 53)
    # print(sp_region)
    small_data = get_region_data(ds4y, small_region)[0:10, :, :]
    print('SMALL: %s' % (small_data.shape,))
    return small_data


def get_dummy():
    '''
    Returns (10, 3, 3) dataset where each 'time series' is 0, 1, ... 9
    '''
    x = np.repeat(np.arange(10), 9)
    return x.reshape(10, 3, 3)


def transpose_region(region_dataset):
    '''
    Instead of [time_series, x, y], work with [x, y, time_series]
    '''
    (series_len, x_len, y_len) = region_dataset.shape

    # 'Fortran-style' will gather time_series points when flattening ('transpose')
    flattened = region_dataset.ravel('F')
    return flattened.reshape(x_len, y_len, series_len)


if __name__ == '__main__':
    t_start = time.time()

    dataset = ds.load_with_len(5840)
    sp = get_sao_paulo_data(dataset)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s' % str(elapsed))
