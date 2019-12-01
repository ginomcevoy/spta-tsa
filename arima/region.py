import logging
import numpy as np
# import os
from collections import namedtuple
import time

from . import dataset as ds
from . import util

Point = namedtuple('Point', 'x y')
Region = namedtuple('Region', 'x1, x2, y1, y2')
TimeInterval = namedtuple('TimeInterval', 't1 t2')


# SMALL_REGION = Region(55, 58, 50, 54)
# SMALL_REGION = Region(0, 1, 0, 1)
SMALL_REGION = Region(0, 3, 0, 4)

# model + test + forecast -> error

# arimitas -> error de cada arima en su entrenamiento
# 1 model + test/region + forecast/region -> error en cada punto


def reshape_1d_to_2d(list_1d, x, y):
    return np.array(list_1d).reshape(x, y)


class SpatialRegion:

    def __init__(self, numpy_dataset):
        self.numpy_dataset = numpy_dataset
        self.log = logging.getLogger()

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
        numpy_region_subset = self.numpy_dataset[region.x1:region.x2, region.y1:region.y2]
        return SpatialRegion(numpy_region_subset)

    def value_at(self, point):
        return self.numpy_dataset[point.x, point.y]

    @property
    def as_array(self):
        '''
        Goes from 2d to 1d
        '''
        (x_len, y_len) = self.numpy_dataset.shape
        return self.numpy_dataset.reshape(x_len * y_len)

    @classmethod
    def create_from_1d(cls, list_1d, x, y):
        '''
        Given a list, reshapes the elements to form a 2d region with (x, y) shape.
        '''
        numpy_dataset = reshape_1d_to_2d(list_1d, x, y)
        return SpatialRegion(numpy_dataset)

    def __str__(self):
        return str(self.numpy_dataset)


class SpatioTemporalRegion(SpatialRegion):

    def series_at(self, point):
        return self.numpy_dataset[point.x, point.y, :]

    def series_len(self):
        return self.numpy_dataset.shape[2]

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[region.x1:region.x2, region.y1:region.y2, :]
        return SpatioTemporalRegion(numpy_region_subset)

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

    @property
    def centroid(self):
        from . import centroid
        centroid_calc = centroid.CalculateCentroid()
        return centroid_calc.find_point_with_least_distance(self)

    @property
    def as_list(self):
        '''
        Returns a list of arrays, where the size of the list is the number of points in the region
        (Region.x * Region.y). Each array is a temporal series.

        Useful when required to iterate each temporal series.
        '''
        return util.spatio_temporal_to_list_of_time_series(self.numpy_dataset)

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

    @classmethod
    def load_1year(cls):

        # load raw data
        pts_1y = ds.POINTS_PER_YEAR
        numpy_dataset = ds.load_with_len(pts_1y)

        # use [x, y, time_series]
        transposed_dataset = transpose_region(numpy_dataset)
        return SpatioTemporalRegion(transposed_dataset)

    @classmethod
    def load_sao_paulo(cls):
        sptr = cls.load_1year()
        sp_region = Region(55, 75, 50, 70)
        sptr_sp = sptr.region_subset(sp_region)

        # we have 4 points per day
        # average these four points to get a smoother curve
        (x_len, y_len) = (sp_region.x2 - sp_region.x1, sp_region.y2 - sp_region.y1)
        new_series_len = int(sptr_sp.series_len() / 4)
        single_point_per_day = np.empty((x_len, y_len, new_series_len))

        for x in range(0, x_len):
            for y in range(0, y_len):
                point = Point(x, y)
                point_series = sptr_sp.series_at(point)
                series_reshape = (new_series_len, 4)
                smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
                # sptr_sp.log.debug('smooth: %s' % smooth)
                single_point_per_day[x, y] = np.array(smooth)

        sptr_sp.log.info('sao paulo: %s' % str(single_point_per_day.shape))
        return SpatioTemporalRegion(single_point_per_day)

    @classmethod
    def copy_series_over_region(cls, series, region_3d):
        (m, n, _) = region_3d.shape
        numpy_dataset = util.copy_array_as_matrix_elements(series, m, n)
        return SpatioTemporalRegion(numpy_dataset)


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

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    sptr = SpatioTemporalRegion.load_4years()
    print(sptr.shape)

    small = sptr.get_small()
    print(small.shape)

    print('centroid %s' % str(small.centroid))

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s' % str(elapsed))
