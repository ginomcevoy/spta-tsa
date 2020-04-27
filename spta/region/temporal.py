import logging
import numpy as np

from spta.dataset import temp_brazil
from spta.util import arrays as arrays_util

from .spatial import SpatialRegion
from . import Point, Region, TimeInterval

# SMALL_REGION = Region(55, 58, 50, 54)
# SMALL_REGION = Region(0, 1, 0, 1)
SAO_PAULO = Region(55, 75, 50, 70)
SMALL_REGION = Region(0, 3, 0, 4)


class SpatioTemporalRegion(SpatialRegion):

    def series_at(self, point):
        return self.numpy_dataset[:, point.x, point.y]

    def series_len(self):
        return self.numpy_dataset.shape[0]

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[:, region.x1:region.x2, region.y1:region.y2]
        return SpatioTemporalRegion(numpy_region_subset)

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        '''
        numpy_region_subset = self.numpy_dataset[ti.t1:ti.t2, :, :]
        return SpatioTemporalRegion(numpy_region_subset)

    def subset(self, region, ti):
        '''
        region: Region
        ti: TimeInterval
        '''
        numpy_region_subset = \
            self.numpy_dataset[ti.t1:ti.t2, region.x1:region.x2, region.y1:region.y2]
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
        return arrays_util.spatio_temporal_to_list_of_time_series(self.numpy_dataset)

    @property
    def as_2d(self):
        '''
        Returns a 2d array, similar to as_list but it is a numpy array.
        Useful when required to iterate each temporal series.
        '''
        return np.array(self.as_list)

    # def get_dummy_region(self):
    #     dummy = Region)

    @classmethod
    def load_4years(cls):

        # load raw data
        numpy_dataset = temp_brazil.load_brazil_temps(4)
        # pts_4y = ds.POINTS_PER_YEAR * 4
        # numpy_dataset = ds.load_with_len(pts_4y)

        # transposed_dataset = transpose_region(numpy_dataset)
        return SpatioTemporalRegion(numpy_dataset)

    @classmethod
    def load_1year(cls):

        # load raw data
        # pts_1y = ds.POINTS_PER_YEAR
        # numpy_dataset = ds.load_with_len(pts_1y)
        numpy_dataset = temp_brazil.load_brazil_temps(1)

        # use [x, y, time_series]
        # transposed_dataset = transpose_region(numpy_dataset)
        logger = logging.getLogger()
        logger.info('Loaded 1year with shape: {}'.format(numpy_dataset.shape))
        return SpatioTemporalRegion(numpy_dataset)

    @classmethod
    def load_1year_last(cls):

        # load raw data
        # pts_1y = ds.POINTS_PER_YEAR
        # numpy_dataset = ds.load_with_len(pts_1y)
        numpy_dataset = temp_brazil.load_brazil_temps_last(1)

        # transposed_dataset = transpose_region(numpy_dataset)
        logger = logging.getLogger()
        logger.info('Loaded last year with shape: {}'.format(numpy_dataset.shape))
        return SpatioTemporalRegion(numpy_dataset)

    @classmethod
    def load_1year_1ppd(cls):
        '''
        Loads the entire dataset, but averages the data within a day (4 points) to a single point.
        '''
        sptr = cls.load_1year()
        _, x_len, y_len = sptr.shape

        # we have 4 points per day
        # average these four points to get a smoother curve
        new_series_len = int(sptr.series_len() / 4)
        single_point_per_day = np.empty((new_series_len, x_len, y_len))

        for x in range(0, x_len):
            for y in range(0, y_len):
                point = Point(x, y)
                point_series = sptr.series_at(point)
                series_reshape = (new_series_len, 4)
                smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
                # sptr.log.debug('smooth: %s' % smooth)
                single_point_per_day[:, x, y] = np.array(smooth)

        sptr.log.info('1year_1ppd: %s' % str(single_point_per_day.shape))
        return SpatioTemporalRegion(single_point_per_day)

    @classmethod
    def load_1year_1ppd_last(cls):
        '''
        Loads the entire dataset, but averages the data within a day (4 points) to a single point.
        '''
        sptr = cls.load_1year_last()
        _, x_len, y_len = sptr.shape

        # we have 4 points per day
        # average these four points to get a smoother curve
        new_series_len = int(sptr.series_len() / 4)
        single_point_per_day = np.empty((new_series_len, x_len, y_len))

        for x in range(0, x_len):
            for y in range(0, y_len):
                point = Point(x, y)
                point_series = sptr.series_at(point)
                series_reshape = (new_series_len, 4)
                smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
                # sptr.log.debug('smooth: %s' % smooth)
                single_point_per_day[:, x, y] = np.array(smooth)

        sptr.log.info('1year_1ppd last: %s' % str(single_point_per_day.shape))
        return SpatioTemporalRegion(single_point_per_day)

    @classmethod
    def load_sao_paulo(cls):
        sptr = cls.load_1year()
        sptr_sp = sptr.region_subset(SAO_PAULO)

        # we have 4 points per day
        # average these four points to get a smoother curve
        (x_len, y_len) = (SAO_PAULO.x2 - SAO_PAULO.x1, SAO_PAULO.y2 - SAO_PAULO.y1)
        new_series_len = int(sptr_sp.series_len() / 4)
        single_point_per_day = np.empty((x_len, y_len, new_series_len))

        for x in range(0, x_len):
            for y in range(0, y_len):
                point = Point(x, y)
                point_series = sptr_sp.series_at(point)
                series_reshape = (new_series_len, 4)
                smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
                # sptr_sp.log.debug('smooth: %s' % smooth)
                single_point_per_day[:, x, y] = np.array(smooth)

        sptr_sp.log.info('sao paulo: %s' % str(single_point_per_day.shape))
        return SpatioTemporalRegion(single_point_per_day)

    @classmethod
    def copy_series_over_region(cls, series, region_3d):
        (_, m, n) = region_3d.shape
        numpy_dataset = arrays_util.copy_array_as_matrix_elements(series, m, n)
        return SpatioTemporalRegion(numpy_dataset)


if __name__ == '__main__':
    import time

    t_start = time.time()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    sptr = SpatioTemporalRegion.load_4years()
    print('brazil: ', sptr.shape)

    small = sptr.get_small()
    print('small: ', small.shape)

    # print('centroid %s' % str(small.centroid))

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: %s' % str(elapsed))
