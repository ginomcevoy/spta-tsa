import logging
import numpy as np

from spta.dataset import temp_brazil
from spta.util import arrays as arrays_util

from .spatial import SpatialRegion, SpatialRegionDecorator, SpatialCluster
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

    def __init__(self, name, region, series_len, ppd, last=True, centroid=None, dataset_dir='raw'):
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

    def __init__(self, numpy_dataset, region_metadata=None):
        super(SpatioTemporalRegion, self).__init__(numpy_dataset, region_metadata)

    def __next__(self):
        '''
        Used for iterating over points.
        The iterator returns the tuple (Point, series) for each point.
        '''
        # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
        if self.point_index >= self.y_len * self.x_len:
            # stop iteration, but allow reuse of iterator from start again
            self.point_index = 0
            raise StopIteration

        # iterate
        point_i_j = Point(int(self.point_index / self.y_len), self.point_index % self.y_len)
        self.point_index += 1
        return (point_i_j, self.series_at(point_i_j))

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

    def repeat_point(self, point):
        '''
        Creates a new spatio-temporal region with this same shape, where all the series are the
        same series as in the provided point.
        '''
        series_at_point = self.series_at(point)
        repeated_series_np = arrays_util.copy_array_as_matrix_elements(series_at_point, self.x_len,
                                                                       self.y_len)
        return SpatioTemporalRegion(repeated_series_np)

    def get_centroid(self, distance_measure=None):
        if self.centroid is not None:
            return self.centroid

        elif distance_measure is None:
            raise ValueError('No pre-calculated centroid, and no distance_measure provided!')

        else:
            # calculate the centroid, ew
            from . import centroid
            centroid_calc = centroid.CalculateCentroid(distance_measure)
            self.centroid = centroid_calc.find_point_with_least_distance(self)
            return self.centroid

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
        spt_region.metadata = sptr_metadata

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


class SpatioTemporalDecorator(SpatialRegionDecorator, SpatioTemporalRegion):
    '''
    A decorator to extend the functionality of a spatio-temporal region.
    Reuses decorator properties from SpatialRegionDecorator.
    '''
    def __init__(self, decorated_region, **kwargs):
        # diamond problem! should use SpatialDecorator constructor, which will account for
        # the metadata parameter in SpatioTemporalCluster
        super(SpatioTemporalDecorator, self).__init__(decorated_region=decorated_region, **kwargs)

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

    def repeat_point(self, point):
        return self.decorated_region.repeat_point(point)

    def get_centroid(self, distance_measure=None):
        return self.decorated_region.get_centroid(distance_measure)

    def __next__(self):
        return self.decorated_region.__next__()


class SpatioTemporalCluster(SpatialCluster, SpatioTemporalDecorator):
    '''
    A subset of a spatio-temporal region that represents a cluster, created by a clustering
    algorithm. A spatio-temporal region may be split into two or more clusters, so that each
    is represented by a label (number) and a mask (spatial region).

    The cluster behaves like a full spatio-temporal region when it comes to iteration and
    and some properties. Specifically:

    - Each cluster retains the shape of the entire region.
    - The iteration of points is made only over the points indicated by the mask (points
        belonging to the cluster).
    - A FunctionRegion can be applied to the cluster, only points in the mask will be considered
        (it uses the iteration above).
    - It can be split along time intervals the same way.
    - The functions as_list and as_2d wiill iterate over the cluster points (since they
        are using the iterator which is overiddenw)
    - Region subsets are not allowed (TODO?), calling region_subset raises NotImplementedError
    - Attempt to use series_at at a point outside of the mask will raise ValueError.
    '''

    def __init__(self, decorated_region, spatial_mask, label=1, region_metadata=None):
        '''
        Creates a new instance.

        numpy_dataset
            numpy array (series_len, x_len, y_len)

        spatial_mask
            a SpatialRegion with values either 1 or 0. A value of 1 in (i, j) indicates that the
            point P(i, j) belongs to this cluster, 0 otherwise

        label
            optional, indicates the i-th member of the output of a clustering algorithm.

        region_metadata
            must be None, different value is not allowed and will raise error (TODO?).
        '''
        # beware the diamond problem!!
        # this should be solved in parents...

        # call parent constructor explicity
        # SpatialCluster.__init__(numpy_dataset, spatial_mask, label, region_metadata)

        super(SpatioTemporalCluster, self).__init__(decorated_region, spatial_mask, label,
                                                    region_metadata)

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        Will create a new spatio-temporal cluster, maintaining current mask and label
        '''
        self.logger.debug('SpatioTemporalCluster interval_subset')
        numpy_region_subset = self.numpy_dataset[ti.t1:ti.t2, :, :]
        return SpatioTemporalCluster(numpy_region_subset, self.spatial_mask, self.label)

    def series_at(self, point):
        '''
        Returns the time series at specified point, the point must belong to cluster mask
        '''
        self.logger.debug('SpatioTemporalCluster {} series_at {}'.format(self.label, point))
        if self.spatial_mask.value_at(point):
            return self.decorated_region.series_at(point)
        else:
            raise ValueError('Point not in cluster mask: {}'.format(point))

    def repeat_point(self, point):
        '''
        Creates a new spatio-temporal cluster with this same shape, where all the series are the
        same series as in the provided point.

        The series is repeated over all points for simplicity, but calling series_at on points
        outside the mask should remain forbidden.
        '''
        self.logger.debug('SpatioTemporalCluster repeat_point')
        repeated_region = self.decorated_region.repeat_point(point)
        return SpatioTemporalCluster(repeated_region, self.spatial_mask, self.label)

    def __next__(self):
        '''
        Used for iterating over points in the cluster. Only points in the mask are iterated!
        The iterator returns the tuple (Point, value) for each point.
        '''

        # find the next point in the mask to iterate
        while True:

            # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
            if self.point_index >= self.y_len * self.x_len:
                # stop iteration, but allow reuse of iterator from start again
                self.point_index = 0
                raise StopIteration

            # candidate point, first is (0, 0)
            i = int(self.point_index / self.y_len)
            j = self.point_index % self.y_len
            point_i_j = Point(i, j)

            # next point to evaluate after this
            self.point_index += 1

            if self.spatial_mask.value_at(point_i_j):
                # found point in the mask
                break

        # return next point in the mask
        return (point_i_j, self.series_at(point_i_j))

    @classmethod
    def from_clustering(cls, spt_region, members, label, centroids=None):
        '''
        Given a clustering result and a label, create a new instance of spatio-temporal cluster.
        A clustering algorithm with parameter k should create k clusters, with labels ranging
        from 0 to k-1.

        The members and label are used to select one of the clusters and to create the
        spatial_mask used as input of a the cluster instance.

        spt_region
            the spatio-temporal region that was clustered

        members
            a 1-d array (not region!) of labels indicating the membership of a point to a cluster,
            given the index position of the point.

        label
            a value i from [0, k-1] that refers to the i-th cluster of a clustering result.

        centroids (optional)
            a 1-d array of indices (not points!) indicating the centroids of each cluster.
            If present, the i-th centroid will be saved as a centroid of the new instance.
        '''
        (_, x_len, y_len) = spt_region.shape

        # sanity checks
        assert members.ndim == 1
        assert members.shape[0] == x_len * y_len

        assert isinstance(label, int)
        assert label >= 0 and label <= np.max(members)

        # build spatial_mask from members and label
        # one way to do this is to select the elements in members that have the label,
        # then reshape the array to 2D using the spt_region shape.

        # ones full of 1, zeros full of 0, then select according to members
        ones = np.repeat(1, x_len * y_len)
        zeros = np.zeros(x_len * y_len, dtype=np.int8)
        mask = np.where(members == label, ones, zeros)

        # 2D SpatialRegion mask
        spatial_mask_np = mask.reshape((x_len, y_len))
        spatial_mask = SpatialRegion(spatial_mask_np)

        cluster = SpatioTemporalCluster(spt_region, spatial_mask, label, None)

        # centroid available?
        if centroids is not None:
            centroid_index = centroids[label]
            i = int(centroid_index / y_len)
            j = centroid_index % y_len
            cluster.centroid = Point(i, j)

        return cluster


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
                                               last=False)
    sp_small = SpatioTemporalRegion.from_metadata(sp_small_md)
    print('sp_small: ', sp_small.shape)

    t_end = time.time()
    elapsed = t_end - t_start

    print('Elapsed: %s' % str(elapsed))
