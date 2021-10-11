import logging
import numpy as np


from spta.util import arrays as arrays_util
from spta.util import fs as fs_util

from . import Region
from .spatial import SpatialDecorator, SpatialCluster, DomainRegion

# SMALL_REGION = Region(55, 58, 50, 54)
# SMALL_REGION = Region(0, 1, 0, 1)
SAO_PAULO = Region(55, 75, 50, 70)
SMALL_REGION = Region(0, 3, 0, 4)


class SpatioTemporalRegion(DomainRegion):
    '''
    A 3-d spatio-temporal region, backed by a 3-d numpy dataset.
    '''

    def __init__(self, numpy_dataset, region_metadata=None):
        super(SpatioTemporalRegion, self).__init__(numpy_dataset)
        self.region_metadata = region_metadata
        self.centroid = None

        # just for simplicity
        self.series_len = self.numpy_dataset.shape[0]

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
            np.repeat(np.nan, repeats=self.series_len)

        return self.numpy_dataset[:, point.x, point.y]

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[:, region.x1:region.x2, region.y1:region.y2]
        return self.new_spatio_temporal_region(numpy_region_subset)

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        '''
        numpy_region_subset = self.numpy_dataset[ti.t1:ti.t2, :, :]
        return self.new_spatio_temporal_region(numpy_region_subset)

    def subset(self, region, ti):
        '''
        region: Region
        ti: TimeInterval
        '''
        numpy_region_subset = \
            self.numpy_dataset[ti.t1:ti.t2, region.x1:region.x2, region.y1:region.y2]
        return self.new_spatio_temporal_region(numpy_region_subset)

    def repeat_series(self, series):
        '''
        Creates a new spatio-temporal region with this same shape, where all the series are
        the same provided series.
        '''
        repeated_series_np = arrays_util.copy_array_as_matrix_elements(series, self.x_len,
                                                                       self.y_len)
        return self.new_spatio_temporal_region(repeated_series_np)

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

    def pickle(self, pickle_home='pickle'):
        '''
        Pickles the object to the file designated by the metadata.
        Raises ValueError if metadata is not available.
        '''
        if self.region_metadata is None:
            raise ValueError('Need metadata to pickle this sptr!')

        # ensure dir
        fs_util.mkdir(self.region_metadata.pickle_dir(pickle_home))

        # delegate
        self.pickle_to(self.region_metadata.pickle_filename(pickle_home))

    def has_scaling(self):
        '''
        Flag indicating that this region cannot be descaled.
        '''
        return False

    def descale(self):
        '''
        This region cannot be descaled, since it was not scaled here. However, we write this
        function here as a sentinel to indicate the error of issuing such a descaling
        request.
        '''
        raise NotImplementedError('Cannot descale a pure SpatioTemporalRegion!')

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

        # keep this too
        self.series_len = decorated_region.series_len

    def region_subset(self, region):
        return self.decorated_region.region_subset(region)

    def series_at(self, point):
        return self.decorated_region.series_at(point)

    def has_centroid(self):
        # if a centroid has been set at parent, honor it
        # TODO we should avoid this?
        if self.centroid is not None:
            return True
        else:
            return self.decorated_region.has_centroid()

    def get_centroid(self, distance_measure=None):
        # if a centroid has been set at parent, honor it
        # TODO we should avoid this?
        if self.centroid is not None:
            return self.centroid
        else:
            return self.decorated_region.get_centroid(distance_measure)

    def save(self):
        return self.decorated_region.save()

    def has_scaling(self):
        '''
        Flag to indicate that the region can be descaled. Ask the decorated region for this
        info. See descale() for details.
        '''
        return self.decorated_region.has_scaling()

    def descale(self):
        '''
        Handle descaling of scaled regions. Since a scaled region can be clustered
        afterwards (or decorated in a different way), we need to be able to handle the
        descaling call at any level of decoration, e.g. in SpatioTemporalCluster.

        We do this by implementing the Chain of Responsibility pattern. By default, the decorated
        region promises to handle the descaling, and the current level will wrap the result
        in a way to keep other functionalities.

        has_scaling() should be called before calling descale(), because trying to descale a
        region that has not been scaled will raise an error.
        '''
        # This approach should work because there is no other data to be saved from a descaled
        # region, except for what we already have in this decorator. Other relevant data should
        # be saved by new_spatio_temporal_region implementation.
        descaled_region = self.decorated_region.descale()
        return self.new_spatio_temporal_region(descaled_region.numpy_dataset)

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

    def new_spatio_temporal_region(self, numpy_dataset):
        '''
        Creates a new instance of SpatioTemporalCluster with the same partition and cluster index.
        This will wrap a new decorated region, that was built by the original decorator
        using Chain of Responsibility pattern.

        Note: new_spatial_region is handled by SpatialCluster
        '''
        new_decorated_region = self.decorated_region.new_spatio_temporal_region(numpy_dataset)
        return SpatioTemporalCluster(decorated_region=new_decorated_region,
                                     partition=self.partition,
                                     cluster_index=self.cluster_index,
                                     region_metadata=self.region_metadata)

    def region_subset(self, region):
        # not allowed for now!
        raise NotImplementedError

    def interval_subset(self, ti):
        '''
        ti: TimeInterval
        Will create a new spatio-temporal cluster with same partition and index
        '''
        # ask the decorated region to do the subset, but this will not create a cluster.
        decorated_interval_subset = self.decorated_region.interval_subset(ti)

        # use new_spatial_region for clusters with the resulting dataset, this will create
        # the appropriate cluster
        subset_cluster = self.new_spatio_temporal_region(decorated_interval_subset.numpy_dataset)

        # drop the metadata, it is not valid anymore...
        # TODO do something smart here?
        subset_cluster.region_metadata = None
        return subset_cluster

    def series_at(self, point):
        '''
        Returns the time series at specified point, the point must belong to cluster
        '''
        # sanity check
        if point is None:
            np.repeat(np.nan, repeats=self.series_len)

        # self.logger.debug('SpatioTemporalCluster {} series_at {}'.format(self.label, point))
        if self.partition.is_member(point, self.cluster_index):
            return self.decorated_region.series_at(point)
        else:
            raise ValueError('Point not in cluster: {}'.format(point))

    @property
    def all_point_indices(self):
        '''
        Returns an array containing all indices in this cluster.
        '''

        # these are all the point indices in the region (not this cluster!)
        all_point_indices_region = np.arange(self.x_len * self.y_len)

        # ask the partition for the membership of these points
        memberships = self.partition.membership_of_point_indices(all_point_indices_region)

        # now filter for this cluster, note that where requires numpy array, and returns 2 tuples
        point_indices_cluster = np.where(np.array(memberships) == self.cluster_index)[0]
        return point_indices_cluster

        # all_point_indices = np.zeros(self.cluster_len, dtype=np.uint32)
        # for i, (point, _) in enumerate(self):
        #     all_point_indices[i] = point.x * self.y_len + point.y

        # return all_point_indices

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

            # the partition knows which cluster(s) the point belongs to
            if self.partition.is_member(candidate_point, self.cluster_index):
                # found a member of the cluster, will be returned
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


if __name__ == '__main__':
    import time

    from spta.dataset.metadata import TemporalMetadata, SamplesPerDay
    from .metadata import SpatioTemporalRegionMetadata

    t_start = time.time()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # small = sptr.get_small()
    # print('small: ', small.shape)

    # print('centroid %s' % str(small.centroid))
    dataset_class_name = 'spta.dataset.csfr.DatasetCSFR'
    temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(4))
    sp_small_md = SpatioTemporalRegionMetadata(name='sp_small',
                                               region=Region(40, 50, 50, 60),
                                               temporal_md=temporal_md,
                                               dataset_class_name=dataset_class_name,
                                               scaled=True)
    sp_small = sp_small_md.create_instance()
    print('sp_small: {} {}'.format(sp_small.shape, sp_small.__class__.__name__))
    print('metadata: {}'.format(sp_small.region_metadata))

    # save as npy
    sp_small.save()

    # save as pickle
    sp_small.pickle()

    # retrieve from pickle
    filename = 'pickle/sp_small_2015_2015_4spd_scaled/sp_small_2015_2015_4spd_scaled.pickle'
    sp_small_pkl = SpatioTemporalRegion.from_pickle(filename)
    assert sp_small.shape == sp_small_pkl.shape

    t_end = time.time()
    elapsed = t_end - t_start

    print('Elapsed: %s' % str(elapsed))
