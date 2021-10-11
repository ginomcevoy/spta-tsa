import os

from . import Point, Region
from .temporal import SpatioTemporalRegion
from .scaling import ScaleFunction
# from spta.dataset import temp_brazil
from spta.dataset import base as dataset_base

from spta.util import log as log_util


class SpatioTemporalRegionMetadata(log_util.LoggerMixin):
    '''
    Metadata for a spatio-temporal region, obtained from a spatio temporal dataset.
    Includes:
        name
            string that identifies the region instance (e.g sp_small)
        region
            an instance of Region with 4 coordinates
        temporal_md
            the temporal metadata for the spatio-temporal region being represented, it can
            indicate a temporal slice or some conversion of the original dataset (e.g. averaging samples)
        dataset_class_name
            a string representing the full path (package.module.class) of the class for
            a dataset, it should be a subclass of spta.dataset.base.FileDataset.
        scaled
            boolean that indicates whether each time series should be scaled to fit in the range
            [0, 1] for each point.
        dataset_dir
            path where to load/store numpy files
        **dataset_kwargs
            optional arguments passed to the dataset constructor, for details see
            spta.dataset.base.FileDataset.

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

    def __init__(self, name, region, temporal_md, dataset_class_name, centroid=None,
                 scaled=True, dataset_dir='raw', **dataset_kwargs):
        self.name = name
        self.region = region
        self.temporal_md = temporal_md
        self.centroid = centroid
        self.scaled = scaled
        self.dataset_dir = dataset_dir

        self.x_len = region.x2 - region.x1
        self.y_len = region.y2 - region.y1

        # get the metadata early
        self.initialize(dataset_class_name, dataset_kwargs)

    def initialize(self, dataset_class_name, dataset_kwargs):
        '''
        Create an instance of a subclass of spta.dataset.base.FileDataset.
        This should not read any data from the filesystem, this is done only
        when requesting an instance of SpatioTemporalRegion.
        '''
        self.dataset = dataset_base.create_dataset_instance(dataset_class_name, **dataset_kwargs)
        self.dataset_temporal_md = self.dataset.dataset_temporal_md

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

        # get (i, j) position relative to the region
        x_region = int(index / self.y_len)
        y_region = index % self.y_len

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
        In order to create the instance, we created an instance of the dataset object,
        using the dataset_class_name string. Here, a temporal slice is retrieved from
        the dataset, and region subset and scaling are performed if needed.
        '''

        # Example of temporal metadata:
        # metadata.TemporalMetadata(2014, 2015, metadata.SamplesPerDay(4)
        numpy_dataset = self.dataset.retrieve(self.temporal_md)

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
            self.logger.debug('Metadata saved: {}'.format(spt_region.region_metadata))

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

        return '{}_{!r}{}'.format(self.name, self.temporal_md, scaled_str)

    def __str__(self):
        return repr(self)
