import logging
import numpy as np

from spta.dataset import temp_brazil
from spta.util import arrays as arrays_util
from spta.util import fs as fs_util

from . import Point, Region
from .spatial import SpatialDecorator, SpatialCluster, DomainRegion

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
            if True, use the last years, else use the first years
        dataset_dir
            path where to load/store numpy files
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

    def __str__(self):
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


class SpatioTemporalRegion(DomainRegion):
    '''
    A 3-d spatio-temporal region, backed by a 3-d numpy dataset.
    '''

    def __init__(self, numpy_dataset, region_metadata=None):
        super(SpatioTemporalRegion, self).__init__(numpy_dataset)
        self.region_metadata = region_metadata
        self.centroid = None

    def __next__(self):
        '''
        Used for iterating over points.
        The iterator returns the tuple (Point, series) for each point.
        '''
        # # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
        # if self.point_index >= self.y_len * self.x_len:
        #     # reached the end, no more points to iterate
        #     # stop iteration, but allow reuse of iterator from start again
        #     self.point_index = 0
        #     raise StopIteration

        # # iterate
        # point_i_j = Point(int(self.point_index / self.y_len), self.point_index % self.y_len)
        # self.point_index += 1
        # return (point_i_j, self.series_at(point_i_j))

        # use the base iterator to get next point
        next_point = super(SpatioTemporalRegion, self).__next__()

        # tuple output
        return (next_point, self.series_at(next_point))

    def series_at(self, point):
        # sanity check
        if point is None:
            series_len, _, _ = self.numpy_dataset.shape
            np.repeat(np.nan, repeats=series_len)

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

    def repeat_series(self, series):
        '''
        Creates a new spatio-temporal region with this same shape, where all the series are
        the same provided series.
        '''
        repeated_series_np = arrays_util.copy_array_as_matrix_elements(series, self.x_len,
                                                                       self.y_len)
        return SpatioTemporalRegion(repeated_series_np)

    def repeat_point(self, point):
        '''
        Creates a new spatio-temporal region with this same shape, where all the series are the
        same series as in the provided point.
        '''
        # reuse repeat_series
        series_at_point = self.series_at(point)
        return self.repeat_series(series_at_point)

    def get_centroid(self, distance_measure=None):
        if self.has_centroid():
            return self.centroid

        elif distance_measure is None:
            raise ValueError('No pre-calculated centroid, and no distance_measure provided!')

        else:
            # calculate the centroid, ugly but works...
            from . import centroid
            centroid_calc = centroid.CalculateCentroid(distance_measure)
            self.centroid, _ = centroid_calc.find_centroid_and_cost(self)
            return self.centroid

    def has_centroid(self):
        '''
        Returrns True iff the centroid has already been calculated.
        '''
        return self.centroid is not None

    def save(self):
        '''
        Saves the dataset to the file designated by the metadata.
        Raises ValueError if metadata is not available.
        '''
        if self.region_metadata is None:
            raise ValueError('Need metadata to save this sptr!')

        # delegate
        self.save_to(self.region_metadata.dataset_filename)

    def pickle(self):
        '''
        Pickles the object to the file designated by the metadata.
        Raises ValueError if metadata is not available.
        '''
        if self.region_metadata is None:
            raise ValueError('Need metadata to pickle this sptr!')

        # ensure dir
        fs_util.mkdir(self.region_metadata.pickle_dir)

        # delegate
        self.pickle_to(self.region_metadata.pickle_filename)

    @property
    def all_point_indices(self):
        '''
        Returns an array containing all indices in this region. Useful when applying some function
        that iterates over all points, but cannot be implemented well with a FunctionRegion.
        Important when restricting the available points, e.g. with clusters.

        Example usage: subsetting the distance_matrix with all points.
        '''
        # just return all possible point indices
        return np.arange(self.x_len * self.y_len)

    @property
    def as_list(self):
        '''
        Returns a list of arrays, where the size of the list is the number of points in the region
        (Region.x * Region.y). Each array is a temporal series.

        Uses the class iterator! Cannot be used inside another class iterator
        '''
        # return arrays_util.spatio_temporal_to_list_of_time_series(self.numpy_dataset)
        return [series for (point, series) in self]

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

    def __str__(self):
        '''
        A string representation: if a name is available, return it.
        Otherwise, return a generic name.
        '''
        if self.region_metadata is not None:
            return self.region_metadata.name
        else:
            return 'SpatioTemporalRegion {}'.format(self.shape)

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

        # subset the data to work only with region
        spt_region = SpatioTemporalRegion(numpy_dataset).region_subset(sptr_metadata.region)

        # save the metadata in the instance, can be useful later
        spt_region.region_metadata = sptr_metadata

        if sptr_metadata.normalized:
            # replace region with normalized version
            spt_region = SpatioTemporalNormalized(spt_region, region_metadata=sptr_metadata)

        logger.info('Loaded dataset {}: {}'.format(sptr_metadata, sptr_metadata.region))
        return spt_region

    @classmethod
    def repeat_series_over_region(cls, series, shape2D):
        '''
        Given a series with length series_len and a 2D shape (x_len, y_len), create a
        SpatioTemporalRegion with shape (series_len, x_len, y_len), where each point contains
        the same provided series.
        '''
        (x_len, y_len) = shape2D
        numpy_dataset = arrays_util.copy_array_as_matrix_elements(series, x_len, y_len)
        return SpatioTemporalRegion(numpy_dataset)


class SpatioTemporalDecorator(SpatialDecorator, SpatioTemporalRegion):
    '''
    A decorator to extend the functionality of a spatio-temporal region.
    Reuses decorator properties from SpatialRegionDecorator.
    '''
    def __init__(self, decorated_region, **kwargs):
        # diamond problem! should use SpatialDecorator constructor, which will account for
        # the metadata parameter in SpatioTemporalCluster
        super(SpatioTemporalDecorator, self).__init__(decorated_region=decorated_region, **kwargs)

        # keep metadata if provided
        self.region_metadata = kwargs.get('region_metadata', None)

    def series_at(self, point):
        return self.decorated_region.series_at(point)

    def series_len(self):
        return self.decorated_region.series_len()

    def region_subset(self, region):
        return self.decorated_region.region_subset(region)

    def interval_subset(self, ti):
        return self.decorated_region.interval_subset(ti)

    def subset(self, region, ti):
        return self.decorated_region.subset(region, ti)

    def repeat_series(self, series):
        return self.decorated_region.repeat_series(series)

    def repeat_point(self, point):
        return self.decorated_region.repeat_point(point)

    def has_centroid(self):
        # if a centroid has been set at parent, honor it
        if self.centroid is not None:
            return True
        else:
            return self.decorated_region.has_centroid()

    def get_centroid(self, distance_measure=None):
        # if a centroid has been set at parent, honor it
        if self.centroid is not None:
            return self.centroid
        else:
            return self.decorated_region.get_centroid(distance_measure)

    def save(self):
        return self.decorated_region.save()

    @property
    def all_point_indices(self):
        return self.decorated_region.all_point_indices


class SpatioTemporalCluster(SpatialCluster, SpatioTemporalDecorator):
    '''
    A subset of a spatio-temporal region that represents a cluster, created by a clustering
    algorithm. A spatio-temporal region may be split into two or more clusters of a partition,
    so that each by a cluster index.

    The cluster behaves similar to a full spatio-temporal region when it comes to iteration and
    and some properties. See SpatialCluster for more information.
    '''

    def __init__(self, decorated_region, partition, cluster_index, region_metadata=None):
        '''
        Creates a new instance.

        decorated_region
            a spatio-temporal region that is being decorated with cluster behavior

        partition
            obtained with a clustering algorithm, it is capable of indicating which points
            belong to a cluster. It knows the number of clusters (k)

        cluster_index
            identifies this cluster in the partition, must be an integer in [0, k-1].

        region_metadata
            allowed for now... not yet useful
        '''
        # beware the diamond problem!!
        # this should be solved in parents...
        super(SpatioTemporalCluster, self).__init__(decorated_region, partition, cluster_index,
                                                    region_metadata=region_metadata)

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        Will create a new spatio-temporal cluster with same partition and index
        '''
        self.logger.debug('{} interval_subset'.format(self))
        decorated_interval_subset = self.decorated_region.interval_subset(ti)

        # we need to clone the partition!
        # if we don't, training and observation will have the same iterators when forecasting
        # TODO is this still true? Probably not...
        return SpatioTemporalCluster(decorated_interval_subset, self.partition.clone(),
                                     self.cluster_index)

    def series_at(self, point):
        '''
        Returns the time series at specified point, the point must belong to cluster
        '''
        # sanity check
        if point is None:
            series_len, _, _ = self.numpy_dataset.shape   # TODO self.series_len
            np.repeat(np.nan, repeats=series_len)

        # self.logger.debug('SpatioTemporalCluster {} series_at {}'.format(self.label, point))
        if self.partition.is_member(point, self.cluster_index):
            return self.decorated_region.series_at(point)
        else:
            raise ValueError('Point not in cluster: {}'.format(point))

    def repeat_series(self, series):
        '''
        Creates a new spatio-temporal cluster with this same shape, where all the series are the
        same as the provided series.

        The series is repeated over all points for simplicity, but calling series_at on points
        outside the cluster should remain forbidden.

        NOTE: no need to reimplement repeat_point, it should correctly use this method
        polymorphically to create a SpatioTemporalCluster.
        '''
        self.logger.debug('{} repeat_series'.format(self))
        repeated_region = self.decorated_region.repeat_series(series)
        return SpatioTemporalCluster(repeated_region, self.partition, self.cluster_index,
                                     self.region_metadata)

    @property
    def all_point_indices(self):
        '''
        Returns an array containing all indices in this region.
        TODO: ask the partition for this, should be faster
        '''
        all_point_indices = np.zeros(self.cluster_len, dtype=np.uint32)
        for i, (point, _) in enumerate(self):
            all_point_indices[i] = point.x * self.y_len + point.y

        return all_point_indices

    def __next__(self):
        '''
        Used for iterating *only* over points in the cluster. Points not in the cluster are skipped
        by this iterator!
        The iterator returns the tuple (Point, series) for each point.
        '''
        while True:

            # use the base iterator to get next candidate point in region
            # when the base iterator stops, we also stop
            try:
                candidate_point = DomainRegion.__next__(self)
            except StopIteration:
                # self.logger.debug('Base region iteration stopped')
                raise

            # self.logger.debug('{}: candidate = {}'.format(self, candidate_point))

            if self.partition.is_member(candidate_point, self.cluster_index):
                # found a member of the cluster
                next_point_in_cluster = candidate_point
                break

            # the point was not in the cluster, try with next candidate

        next_value = self.series_at(next_point_in_cluster)
        return (next_point_in_cluster, next_value)

    def __str__(self):
        '''
        A string representation: if a name is available, return it.
        Otherwise, return a generic name.
        '''
        if hasattr(self, 'name'):
            return self.name
        else:
            return 'SpatioTemporalCluster {}'.format(self.shape)

    @classmethod
    def from_fuzzy_clustering(cls, spt_region, uij, cluster_index, threshold, centroids=None):
        '''
        Similar to from_crisp_clustering, but for fuzzy clustering results.
        uij is a 2-d membership array, threshold defines how the cluster can consider point
        membership depending on its fuzzy membership values.

        TODO should be handled by PartitionRegionFuzzy instead!
        '''
        # (_, x_len, y_len) = spt_region.shape

        # (N, k) = uij.shape
        # assert N == x_len * y_len

        # assert isinstance(cluster_index, int)
        # assert cluster_index >= 0 and cluster_index <= k

        # # build mask_region from uij, cluster_index and threshold
        # mask_region = MaskRegionFuzzy.from_uij_and_region(uij, x_len, y_len, cluster_index,
        #                                                   threshold)
        # mask_region.name = '{}-MaskFuzzy{}'.format(spt_region, cluster_index)

        # # build one cluster for this cluster_index
        # cluster = SpatioTemporalCluster(spt_region, mask_region, None)
        # cluster.name = '{}-fcluster{}'.format(spt_region, cluster_index)

        # # centroid available?
        # if centroids is not None:
        #     centroid_index = centroids[cluster_index]
        #     i = int(centroid_index / y_len)
        #     j = centroid_index % y_len
        #     cluster.centroid = Point(i, j)

        # return cluster
        raise NotImplementedError


class SpatioTemporalNormalized(SpatioTemporalDecorator):
    '''
    A decorator that normalizes the temporal series by scaling all the series to values
    between 0 and 1. This can be achieved by applying the formula to each series:

    normalized_series = (series - min) / (max - min)

    We store the min and max values, so that the series can be later recovered by using:

    series = normalized_series (max - min) + min

    The min and max values are valid for each series. These are stored in spatial regions
    called normalization_min and normalization_max, respectively.

    If min = max, then scaled is fixed to 0.
    If the series is Nan, save the  series, and set min = max = 0.
    '''
    def __init__(self, decorated_region, **kwargs):

        self.logger.debug('SpatioTemporalNormalized kwargs: {}'.format(kwargs))

        (series_len, x_len, y_len) = decorated_region.shape

        # moved here to avoid circular imports between FunctionRegion and SpatioTemporalRegion
        from . import function as function_region

        # calculate the min, max values for each series
        # we will save these outputs to allow denormalization
        minFunction = function_region.FunctionRegionScalarSame(np.nanmin, x_len, y_len)
        self.normalization_min = minFunction.apply_to(decorated_region)

        maxFunction = function_region.FunctionRegionScalarSame(np.nanmax, x_len, y_len)
        self.normalization_max = maxFunction.apply_to(decorated_region)

        # this function normalizes the series at each point independently of other points
        def normalize(series):

            # calculate min/max (again...)
            series_min = np.nanmin(series)
            series_max = np.nanmax(series)

            # sanity checks
            if np.isnan(series_min) or np.isnan(series_max):
                return np.repeat(np.nan, repeats=series_len)

            if series_min == series_max:
                return np.zeros(series_len)

            # normalize here
            return (series - series_min) / (series_max - series_min)

        # call the function
        normalizing_function = function_region.FunctionRegionSeriesSame(normalize, x_len, y_len)
        normalized_region = normalizing_function.apply_to(decorated_region, series_len)
        normalized_region.region_metadata = decorated_region.region_metadata

        # use the normalized region here, all decorating functions from other decorators should be
        # applied to this normalized region instead
        super(SpatioTemporalNormalized, self).__init__(normalized_region, **kwargs)

    def save(self):
        '''
        In addition of saving the numpy dataset, also save the min/max regions.
        Requires the metadata!
        '''

        # save dataset
        super(SpatioTemporalNormalized, self).save()

        # save min
        min_filename = self.region_metadata.norm_min_filename
        np.save(min_filename, self.normalization_min.numpy_dataset)
        self.logger.info('Saved norm_min to {}'.format(min_filename))

        # save max
        max_filename = self.region_metadata.norm_max_filename
        np.save(max_filename, self.normalization_max.numpy_dataset)
        self.logger.info('Saved norm_max to {}'.format(max_filename))

    def __next__(self):
        '''
        Don't use the default iterator here, which comes from SpatialDecorator.
        Instead, iterate like a spatio-temporal region, as the decorated region does.
        '''
        return self.decorated_region.__next__()


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


if __name__ == '__main__':
    import time

    t_start = time.time()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # small = sptr.get_small()
    # print('small: ', small.shape)

    # print('centroid %s' % str(small.centroid))
    sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4,
                                               last=False, normalized=True)
    sp_small = SpatioTemporalRegion.from_metadata(sp_small_md)
    print('sp_small: ', sp_small.shape)

    # save as npy
    sp_small.save()

    # save as pickle
    sp_small.pickle()

    # retrieve from pickle
    sp_small_pkl = SpatioTemporalRegion.from_pickle('pickle/sp_small_1y_4ppd_first_norm.pickle')
    assert sp_small.shape == sp_small_pkl.shape

    t_end = time.time()
    elapsed = t_end - t_start

    print('Elapsed: %s' % str(elapsed))
