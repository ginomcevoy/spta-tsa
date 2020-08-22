import os

from . import Point, Region
from .temporal import SpatioTemporalRegion
from .scaling import ScaleFunction
from spta.dataset import temp_brazil

from spta.util import log as log_util


class SpatioTemporalRegionMetadata(log_util.LoggerMixin):
    '''
    Metadata for a spatio-temporal region, obtained from a spatio temporal dataset.
    Includes:
        name
            string that identifies the region instance (e.g sp_small)
        region
            an instance of Region with 4 coordinates
        year_start
            the starting year (always work with whole years)
        year_end
            the ending year, if it is the same as year_start then that year is used.
        spd
            the samples per day. By default we assume that the dataset has 4 samples per day
            (current dataset being used).
            TODO improve this representation when other datasets are used
        scaled
            boolean that indicates whether each time series should be scaled to fit in the range
            [0, 1] for each point.
        dataset_dir
            path where to load/store numpy files

    Examples of representation:
    sp_small_2014_2014_1spd_scaled
        - Region called "sp_small".
        - Uses the dataset for the year 2014 (365 days).
        - Use 1 sample per day. Since the dataset uses 4 samples per day, these 4 samples will be
          averaged into a single value per day, for a total of 365 samples per point.
        - Scale the dataset, see ScaleFunction for detail.s

    whole_brazil_2014_2015_4spd
        - Region called "whole_brazil".
        - Uses the dataset for the years 2014 and 2015 (365 * 2 days).
        - Uses 4 samples per day. Since the dataset already uses 4 samples per day, no averaging
          is done.
        - No scaling is applied.

    TODO break the assumption that the dataset has 4 points per day
    TODO break the assumption that the dataset goes up to the end of 2015
    TODO support other distance measures
    '''

    def __init__(self, name, region, year_start, year_end, spd, centroid=None,
                 scaled=True, dataset_dir='raw'):
        self.name = name
        self.region = region
        self.year_start = year_start
        self.year_end = year_end
        self.spd = spd
        self.scaled = scaled
        self.dataset_dir = dataset_dir

    @property
    def dataset_filename(self):
        '''
        Ex 'raw/sp_small_2015_2015_4spd_scaled.npy'
        '''
        return '{}/{}.npy'.format(self.dataset_dir, self)

    @property
    def distances_filename(self):
        '''
        Ex raw/distances_sp_small_2015_2015_4spd_scaled.npy'
        '''
        return '{}/distances_{}.npy'.format(self.dataset_dir, self)

    @property
    def scaled_min_filename(self):
        '''
        Ex 'raw/sp_small_2015_2015_4spd_min.npy'
        '''
        return '{}/{}_min.npy'.format(self.dataset_dir, self)

    @property
    def scaled_max_filename(self):
        '''
        Ex 'raw/sp_small_2015_2015_4spd_max.npy'
        '''
        return '{}/{}_max.npy'.format(self.dataset_dir, self)

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

    def create_instance(self):
        '''
        Creates an instance of SpatioTemporalRegion using the current metadata.
        Currently supports only 1y and 4y, 1spd and 4spd.

        Assumes temp_brazil dataset!
        Assumes spd = 1 or spd = 4!
        '''
        # big assumption
        assert self.spd == 1 or self.spd == 4

        # read the dataset according to the year interval and spd
        numpy_dataset = temp_brazil.retrieve_dataset_interval(year_start=self.year_start,
                                                              year_end=self.year_end,
                                                              spd=self.spd)
        # subset the data to work only with region
        spt_region = SpatioTemporalRegion(numpy_dataset).region_subset(self.region)

        # save the metadata in the instance, can be useful later
        spt_region.region_metadata = self

        if self.scaled:
            # replace region with scaled version
            series_len, x_len, y_len = spt_region.shape
            scale_function = ScaleFunction(x_len, y_len)
            spt_region = scale_function.apply_to(spt_region, series_len)
            self.logger.debug('Scaling data: {}'.format(spt_region.shape))

        self.logger.info('Loaded spatio-temporal region {}: {}'.format(self, self.region))
        return spt_region

    def output_dir(self, output_home):
        '''
        Directory to store outputs relevant to this region metadata.
        '''
        return os.path.join(output_home, repr(self))

    def pickle_dir(self, pickle_home='pickle'):
        '''
        Directory to store pickle files relevant to this region metadata.
        '''
        return os.path.join(pickle_home, repr(self))

    def pickle_filename(self, pickle_home='pickle'):
        '''
        Ex 'pickle/sp_small_2015_2015_4spd_scaled/sp_small_2015_2015_4spd_scaled.pickle'
        '''
        pickle_dir = self.pickle_dir(pickle_home)
        return '{}/{!r}.pickle'.format(pickle_dir, self)

    def __repr__(self):
        '''
        Ex sp_small_2014_2015_1spd_scaled'
        '''
        scaled_str = ''
        if self.scaled:
            scaled_str = '_scaled'

        return '{}_{}_{}_{}spd{}'.format(self.name, self.year_start, self.year_end, self.spd,
                                         scaled_str)

    def __str__(self):
        return repr(self)
