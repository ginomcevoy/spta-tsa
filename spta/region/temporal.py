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


class SpatioTemporalRegionMetadata(object):
    '''
    Metadata for a spatio temporal region of a spatio temporal dataset.
    Includes:
        name
            string (e.g sp_small)
        region
            a 2D region
        series_len
            the length of the temporal series
        ppd
            the points per day
        last
            if True, use the last years, else use the first years (TODO add this to the names)
        dataset_dir
            path where to load/store numpy files
    '''

    def __init__(self, name, region, series_len, ppd, last=True, dataset_dir='raw'):
        self.name = name
        self.region = region
        self.series_len = series_len
        self.ppd = ppd
        self.last = last
        self.dataset_dir = dataset_dir

    @property
    def years(self):
        '''
        Integer representing number of years of series length
        '''
        days = self.series_len / self.ppd
        return int(days / 365)

    @property
    def time_str(self):
        '''
        A string representing the series length in days.
        For now assume that we are always using entire years.
        Ex: series_len = 365 and ppd = 1 -> time_str = 1y
        Ex: series_len = 730 and ppd = 1 -> time_str = 2y
        Ex: series_len = 1460 and ppd = 4 -> time_str = 1y
        '''
        return '{}y'.format(self.years)

    @property
    def dataset_filename(self):
        '''
        Ex 'raw/sp_small_1y_4ppd.npy'
        '''
        return '{}/{}.npy'.format(self.dataset_dir, self)

    @property
    def distances_filename(self):
        '''
        Ex raw/distances_sp_small_1y_4ppd.npy'
        '''
        return '{}/distances_{}.npy'.format(self.dataset_dir, self)

    def __str__(self):
        '''
        Ex sp_small_1y_4ppd'
        '''
        return '{}_{}_{}ppd'.format(self.name, self.time_str, self.ppd)


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

    @property
    def shape_2d(self):
        _, x_len, y_len = self.shape
        return (x_len, y_len)

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
    def from_metadata(cls, sptr_metadata):
        '''
        Loads the data of a spatio temporal region given its metadata
        (SpatioTemporalRegionMetadata).
        Currently supports only 1y and 4y, 1ppd and 4ppd.
        '''

        # big assumption
        assert sptr_metadata.ppd == 1 or sptr_metadata.ppd == 4

        logger = logging.getLogger()

        # read the dataset according to the number of years and start/finish of dataset
        # default is 4ppd...
        if sptr_metadata.last:
            numpy_dataset = temp_brazil.load_brazil_temps_last(sptr_metadata.years)
        else:
            numpy_dataset = temp_brazil.load_brazil_temps(sptr_metadata.years)

        # convert to 1ppd?
        if sptr_metadata.ppd == 1:
            numpy_dataset = average_4ppd_to_1ppd(numpy_dataset, logger)

        logger.info('Loaded dataset: {}'.format(sptr_metadata))
        return SpatioTemporalRegion(numpy_dataset).region_subset(sptr_metadata.region)

    @classmethod
    def copy_series_over_region(cls, series, region_3d):
        (_, m, n) = region_3d.shape
        numpy_dataset = arrays_util.copy_array_as_matrix_elements(series, m, n)
        return SpatioTemporalRegion(numpy_dataset)


def average_4ppd_to_1ppd(sptr_numpy, logger=None):
    '''
    Given a spatio temporal region with the defaults of 4 points per day (ppd=4), average the
    points in each day to get 1 point per day(ppd = 1)
    '''
    (series_len, x_len, y_len) = sptr_numpy.shape

    # we have 4 points per day
    # average these four points to get a smoother curve
    new_series_len = int(series_len / 4)
    single_point_per_day = np.empty((new_series_len, x_len, y_len))

    for x in range(0, x_len):
        for y in range(0, y_len):
            point_series = sptr_numpy[:, x, y]
            series_reshape = (new_series_len, 4)
            smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
            # sptr.log.debug('smooth: %s' % smooth)
            single_point_per_day[:, x, y] = np.array(smooth)

    if logger:
        log_msg = 'reshaped 4ppd {} to 1ppd {}'
        logger.info(log_msg.format(sptr_numpy.shape, single_point_per_day.shape))


if __name__ == '__main__':
    import time

    t_start = time.time()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # sptr = SpatioTemporalRegion.load_4years()
    # print('brazil: ', sptr.shape)

    # small = sptr.get_small()
    # print('small: ', small.shape)

    # print('centroid %s' % str(small.centroid))

    # use metadata... beware of circular imports
    from spta.metadata import SpatioTemporalRegionMetadata
    sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4,
                                               last=False)
    sp_small = SpatioTemporalRegion.from_metadata(sp_small_md)
    print('sp_small: ', sp_small.shape)

    t_end = time.time()
    elapsed = t_end - t_start

    print('Elapsed: %s' % str(elapsed))
