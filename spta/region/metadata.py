import numpy as np

from . import Point, Region
from .temporal import SpatioTemporalRegion
from .scaling import ScaleFunction
from spta.dataset import temp_brazil

from spta.util import log as log_util


class SpatioTemporalRegionMetadata(log_util.LoggerMixin):
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
            if True, use the last years, else use the first years
        dataset_dir
            path where to load/store numpy files

    TODO support other distance measures
    '''

    def __init__(self, name, region, series_len, ppd, last=True, centroid=None,
                 normalized=True, dataset_dir='raw', pickle_dir='pickle'):
        self.name = name
        self.region = region
        self.series_len = series_len
        self.ppd = ppd
        self.last = last
        self.normalized = normalized
        self.dataset_dir = dataset_dir
        self.pickle_dir = pickle_dir

    def index_to_absolute_point(self, index):
        '''
        Given a 2d index, recover the original Point coordinates.
        This is useful when applied to a medoid index, because it will give the medoid position
        in the original dataset.

        Assumes that the medoid index has been calculated from the region specified in this
        metadata instance.
        '''
        y_len = self.region.y2 - self.region.y1

        # get (i, j) position relative to the region
        x_region = int(index / y_len)
        y_region = index % y_len

        # add region offset like this
        return self.absolute_position_of_point(Point(x_region, y_region))

    def absolute_position_of_point(self, point):
        '''
        Given a point, recover its original coordinates.
        Assumes that the provided point has been calculated from the region specified in this
        metadata instance.
        '''
        # get the region offset and add to point
        x_offset, y_offset = self.region.x1, self.region.y1
        return Point(point.x + x_offset, point.y + y_offset)

    def absolute_coordinates_of_region(self, region):
        '''
        Given a rectangle region, recover its original coordinates.
        Assumes that the provided points have been calculated from the region specified in this
        metadata instance.
        '''
        # get the offset of (x1, y1) and (x2, y2)
        corner1 = Point(region.x1, region.y1)
        corner1_absolute = self.absolute_position_of_point(corner1)
        corner2 = Point(region.x2, region.y2)
        corner2_absolute = self.absolute_position_of_point(corner2)

        # build a new Region with the absolute coordinates
        return Region(corner1_absolute.x, corner2_absolute.x,
                      corner1_absolute.y, corner2_absolute.y)

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
        Ex 'raw/sp_small_1y_4ppd_norm.npy'
        '''
        return '{}/{}.npy'.format(self.dataset_dir, self)

    @property
    def distances_filename(self):
        '''
        Ex raw/distances_sp_small_1y_4ppd_norm.npy'
        '''
        return '{}/distances_{}.npy'.format(self.dataset_dir, self)

    @property
    def norm_min_filename(self):
        '''
        Ex 'raw/sp_small_1y_4ppd_min.npy'
        '''
        return '{}/{}_min.npy'.format(self.dataset_dir, self)

    @property
    def norm_max_filename(self):
        '''
        Ex 'raw/sp_small_1y_4ppd_max.npy'
        '''
        return '{}/{}_max.npy'.format(self.dataset_dir, self)

    @property
    def pickle_filename(self):
        '''
        Ex 'pickle/sp_small_1y_4ppd_norm.pickle'
        '''
        return '{}/{}.pickle'.format(self.pickle_dir, self)

    def create_instance(self):
        '''
        Creates an instance of SpatioTemporalRegion using the current metadata.
        Currently supports only 1y and 4y, 1ppd and 4ppd.

        Assumes temp_brazil dataset!
        '''

        # big assumption
        assert self.ppd == 1 or self.ppd == 4

        # read the dataset according to the number of years and start/finish of dataset
        # default is 4ppd...
        if self.last:
            numpy_dataset = temp_brazil.load_brazil_temps_last(self.years)
        else:
            numpy_dataset = temp_brazil.load_brazil_temps(self.years)

        # convert to 1ppd?
        if self.ppd == 1:
            numpy_dataset = average_4ppd_to_1ppd(numpy_dataset, self.logger)

        # subset the data to work only with region
        spt_region = SpatioTemporalRegion(numpy_dataset).region_subset(self.region)

        # save the metadata in the instance, can be useful later
        spt_region.region_metadata = self

        if self.normalized:
            # replace region with scaled version
            # TODO change the variable name
            series_len, x_len, y_len = spt_region.shape
            scale_function = ScaleFunction(x_len, y_len)
            spt_region = scale_function.apply_to(spt_region, series_len)

        self.logger.info('Loaded dataset {}: {}'.format(self, self.region))
        return spt_region

    def __repr__(self):
        '''
        Ex sp_small_1y_4ppd_norm'
        '''
        norm_str = ''
        if self.normalized:
            norm_str = '_norm'

        last_str = ''
        if not self.last:
            last_str = '_first'

        return '{}_{}_{}ppd{}{}'.format(self.name, self.time_str, self.ppd, last_str, norm_str)

    def __str__(self):
        return repr(self)


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

    return single_point_per_day
