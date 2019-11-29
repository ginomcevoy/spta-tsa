# import h5py
# import numpy as np
# import os
from collections import namedtuple
import time

from . import dataset as ds


Region = namedtuple('Region', 'x1, x2, y1, y2')
TimeInterval = namedtuple('TimeInterval', 't1 t2')


SMALL_REGION = Region(55, 58, 50, 53)


class SpatioTemporalRegion:

    def __init__(self, numpy_dataset):
        self.numpy_dataset = numpy_dataset

    @property
    def as_numpy(self):
        return self.numpy_dataset

    @property
    def shape(self):
        return self.numpy_dataset.shape

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[region.x1:region.x2, region.y1:region.y2, :]
        return SpatioTemporalRegion(numpy_region_subset)

    def series_len(self):
        return self.numpy_dataset.shape[2]

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        '''
        numpy_region_subset = self.numpy_dataset[:, :, ti.t1:ti.t2]
        return SpatioTemporalRegion(numpy_region_subset)

    def subset(self, region, ti):
        '''
        region: Region
        ti: TimeInterval
        '''
        numpy_region_subset = \
            self.numpy_dataset[region.x1:region.x2, region.y1:region.y2, ti.t1:ti.t2]
        return SpatioTemporalRegion(numpy_region_subset)

    def get_small(self):
        return self.region_subset(SMALL_REGION)

    def get_dummy(self):
        small_interval = TimeInterval(0, 10)
        return self.subset(SMALL_REGION, small_interval)

    # def get_dummy_region(self):
    #     dummy = Region)

    @classmethod
    def load_4years(cls):

        # load raw data
        pts_4y = ds.POINTS_PER_YEAR * 4
        numpy_dataset = ds.load_with_len(pts_4y)

        # use [x, y, time_series]
        transposed_dataset = transpose_region(numpy_dataset)
        return SpatioTemporalRegion(transposed_dataset)


def transpose_region(numpy_region_dataset):
    '''
    Instead of [time_series, x, y], work with [x, y, time_series]
    '''
    (series_len, x_len, y_len) = numpy_region_dataset.shape

    # 'Fortran-style' will gather time_series points when flattening ('transpose')
    flattened = numpy_region_dataset.ravel('F')
    return flattened.reshape(x_len, y_len, series_len)


if __name__ == '__main__':
    t_start = time.time()

    sptr = SpatioTemporalRegion.load_4years()
    print(sptr.shape)

    small = sptr.get_small()
    print(small.shape)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s' % str(elapsed))
