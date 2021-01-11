import numpy as np
import os
import pickle

from . import Point
from .base import BaseRegion
from .spatial import SpatialRegion, SpatialCluster
from .temporal import SpatioTemporalCluster

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import arrays as arrays_util


class PartitionRegion(BaseRegion):
    '''
    A 2-d region that is represents the result of applying a clustering algorithm on a spatial or
    spatio-temporal region. A partition is made of a mask indicating membership to a cluster, and
    can be used to create the cluster instances.

    Given a partition, each cluster can be identified by the partition and a cluster index, ranging
    from 0 to k-1, where k is the number of clusters specified in the clustering agorithm.
    All clusters created by a clustering algorithm are attached to the same partition.

    The numpy_dataset remains opaque here, subclasses should determine the nature of the dataset.
    Since this is a region, x_len and y_len are available as second-to-last and last dimensions.

    These are the functionalities of the partition:

    (1) is_member(point, index):
        Indicates whether a point in the region belongs to a cluster given its index.
        This can be used by clusters to support iteration over their own points.
        Note that the partition does not handle the iteration directly, the cluster is responsible
        for its own iteration and queries the partition for membership.

    (2) membership_of_points(points):
        Find the membership of a list of points. This returns a list, where each element is the
        cluster index for that point (value within [0, k-1]).
        Assumes that the point coordinates are relative to the region specified in the partition.

    (3) membership_of_point_indices(point_indices):
        Same as (2) but for point indices instead of Point instances. This is faster if done with
        numpy operations, so (2) uses (3) internally.

    (4) cluster_size(index):
        Returns the size of a cluster (number of points) given its index. By default, iterates
        over all the points and asks for membership (slow?)

    (5) create_spatial_cluster(spatial_region, index):
        Returns an instance of SpatialCluster for this partition and given index.

    (6) create_spt_cluster(spt_region, index):
        Returns an instance of SpatioTemporalCluster for this partition and given index.

    (7) to_pickle(path):
        Saves this instance as a pickle object, can later be retrieved using try_from_pickle(path).
    '''

    def __init__(self, numpy_dataset, k):
        '''
        Creates an instance, the value of k (number of clusters in partition) must be known.
        '''
        super(PartitionRegion, self).__init__(numpy_dataset)
        self.k = k

    def is_member(self, point, cluster_index):
        '''
        Indicates whether a point in the region belongs to a cluster given its index.
        This can be used by clusters to support iteration over their own points.
        Note that the partition does not handle the iteration directly, the cluster is responsible
        for its own iteration and queries the partition for membership.

        Subclasses must override this.
        '''
        raise NotImplementedError

    def membership_of_points(self, points):
        '''
        Find the membership of a list of points. This returns a list, where each element is the
        cluster index for that point (value within [0, k-1]).
        Assumes that the point coordinates are relative to the region specified in the partition.

        Uses membership_of_indices implementation
        '''
        point_indices = [
            point.x * self.y_len + point.y
            for point
            in points
        ]

        return self.membership_of_point_indices(point_indices)

    def membership_of_point_indices(self, point_indices):
        '''
        Same as membership_of_points(points) but for point indices instead of Point instances.
        This is faster if done with numpy operations, so membership_of_points(points) uses this
        method internally.

        Subclasses must override this.
        '''
        raise NotImplementedError

    def cluster_len(self, cluster_index):
        '''
        Returns the size of a cluster (number of points) given its index. By default, iterates
        over all the points and asks for membership...
        '''

        members = 0
        for point in self:
            if self.is_member(point, cluster_index):
                members += 0
        return members

    def clone(self):
        '''
        Return an identical partition instance. TODO not needed anymore?
        '''
        raise NotImplementedError

    def create_spatial_cluster(self, spatial_region, cluster_index):
        '''
        Returns an instance of SpatialCluster for this partition and given index.
        '''
        # sanity checks
        x_len_msg = 'Expected x_len: {}, partition got x_len: {}'.format(spatial_region.x_len, self.x_len)
        assert self.x_len == spatial_region.x_len, x_len_msg

        y_len_msg = 'Expected y_len: {}, partition got y_len: {}'.format(spatial_region.y_len, self.y_len)
        assert self.y_len == spatial_region.y_len, y_len_msg

        return SpatialCluster(spatial_region, self, cluster_index)

    def create_all_spatial_clusters(self, spatial_region):
        '''
        Returns an array of spatial clusters, calls create_spatial_cluster to create each
        one, looping over the indices [0, k-1].
        '''
        clusters = []
        for i in range(0, self.k):
            cluster_i = self.create_spatial_cluster(spatial_region, i)
            clusters.append(cluster_i)

        return clusters

    def create_spt_cluster(self, spt_region, cluster_index):
        '''
        Returns an instance of SpatioTemporalCluster for this partition and given index.
        '''
        # sanity checks
        x_len_msg = 'Expected x_len: {}, partition got x_len: {}'.format(spt_region.x_len, self.x_len)
        assert self.x_len == spt_region.x_len, x_len_msg

        y_len_msg = 'Expected y_len: {}, partition got y_len: {}'.format(spt_region.y_len, self.y_len)
        assert self.y_len == spt_region.y_len, y_len_msg

        return SpatioTemporalCluster(spt_region, self, cluster_index, spt_region.region_metadata)

    def create_all_spt_clusters(self, spt_region, centroid_indices=None, medoids=None):
        '''
        Returns an array of spatio-temporal clusters, calls create_spt_cluster to create each
        one, looping over the indices [0, k-1].

        centroid_indices:
            a 1-d array of indices (not points!) indicating the centroids of each cluster.
            If present, the i-th centroid will be saved as a centroid of the new instance.
            Cannot be used with medoids.

        medoids
            an array of medoid point instances to use as centroids. Cannot be used with
            centroid_indices

        TODO omit medoids argument if partition already has medoids in it
        '''

        # only one
        assert centroid_indices is None or medoids is None

        clusters = []
        for i in range(0, self.k):
            cluster_i = self.create_spt_cluster(spt_region, i)
            cluster_i.name = 'cluster{}'.format(i)

            # centroids available?
            if centroid_indices:
                centroid_index = centroid_indices[i]
                ci = int(centroid_index / self.y_len)
                cj = centroid_index % self.y_len
                centroid_i = Point(ci, cj)
                cluster_i.centroid = centroid_i

            # medoids available?
            if medoids:
                cluster_i.centroid = medoids[i]

            clusters.append(cluster_i)

        return clusters

    def to_pickle(self, pickle_full_path):
        '''
        Saves this partition as a pickle object to the given file path, overwrites the file.
        The directory containing the file is created if it does not exist.
        '''

        # create parent dir
        fs_util.mkdir(os.path.dirname(pickle_full_path))

        # call pickle
        with open(pickle_full_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
            self.logger.debug('Saved partition with k={} at {}'.format(self.k, pickle_full_path))

    @classmethod
    def try_from_pickle(cls, pickle_full_path):
        '''
        Tries to load a partition via pickle given a file path, and returns the instance.
        Raises an excepti-on if the pickle fails, e.g. if the file does not exist.
        '''
        logger = log_util.LoggerMixin().logger
        logger.debug('Attempting to load cluster partition at {}'.format(pickle_full_path))
        partition = None
        with open(pickle_full_path, 'rb') as pickle_file:
            partition = pickle.load(pickle_file)

        return partition


class PartitionRegionCrisp(PartitionRegion):
    '''
    A partition that is the result of applying a 'crisp' (non-fuzzy) clustering algorithm.

    It contains a 2-d array that indicates the membership of each point, using 2-d region
    coordinates. The value indicates the index of the cluster to which the point belongs to.

    Has these additional functionalities

    (5) merge_clusters_2d(spatial_clusters):
        Given a list of spatial clusters, which are compatible with this partition (same
        dimensions, valid values at the member points), create a single spatial region where the
        value at each point corresponds to the value at the cluster for which the point is
        a member.

    (6) merge_with_representatives(spatial_clusters, representatives):
        Similarly to merge_clusters_2d, returns a spatial region. However, the value at each point
        is the value from the specified representative of a cluster, to which the point belongs.
    '''

    def __init__(self, numpy_dataset, k):
        # assume that the numpy_dataset is a 2-d array (membership matrix)
        assert numpy_dataset.ndim == 2

        # sanity check on membership: should be k different labels present
        k_dataset = int(np.max(numpy_dataset)) + 1
        assert k == k_dataset

        super(PartitionRegionCrisp, self).__init__(numpy_dataset, k)

    def is_member(self, point, cluster_index):
        '''
        Returns True iff the point is a member of the cluster with the specified index.
        '''
        # sanity check
        if point is None:
            return False

        return self.numpy_dataset[point.x, point.y] == cluster_index

    def membership_of_point_indices(self, point_indices):
        '''
        Same as membership_of_points(points) but for point indices instead of Point instances.
        This is faster if done with numpy operations, so membership_of_points(points) uses this
        method internally.

        '''
        # Works as expected, an array is returned where each value is the value of the mask at the
        # corresponding index. This should be faster than a loop, right?
        return self.numpy_dataset.take(point_indices).tolist()

    def cluster_len(self, cluster_index):
        '''
        Returns the size of a cluster (number of points) given its index.
        Reimplemented for efficiency here.
        '''
        return np.count_nonzero(self.numpy_dataset == cluster_index)

    def clone(self):
        return PartitionRegionCrisp(np.copy(self.numpy_dataset), self.k)

    def merge_clusters_2d(self, spatial_clusters):
        '''
        Given a list of spatial clusters, which are compatible with this partition (same
        dimensions, valid values at the member points), create a single spatial region where the
        value at each point corresponds to the value at the cluster for which the point is
        a member.

        Expects k clusters, each with a different cluster_index in [0, k-1].
        '''
        # sanity checks: same region shape
        for cluster in spatial_clusters:
            assert cluster.x_len == self.x_len
            assert cluster.y_len == self.y_len

        # order clusters by their index
        sorted(spatial_clusters, key=lambda cluster: cluster.cluster_index)

        # sanity checks: there are k clusters, and all indices are present
        indices = [
            cluster.cluster_index
            for cluster
            in spatial_clusters
        ]
        if indices != list(range(0, self.k)):
            error_msg = 'Bad cluster merge: expected {}, got {}'
            raise ValueError(error_msg.format(list(range(0, self.k)), indices))

        # assuming 2d dataset
        dtype = spatial_clusters[0].as_numpy.dtype
        merged_dataset = np.empty((self.x_len, self.y_len), dtype=dtype)

        # this will iterate all the points in the region (not over clusters)!
        for point in self:

            # get the cluster that contains this point, and extract its value
            # this leverages the fact that the clusters are sorted!
            point_index = int(self.numpy_dataset[point.x, point.y])
            value_at_point = spatial_clusters[point_index].value_at(point)

            merged_dataset[point.x, point.y] = value_at_point

        return SpatialRegion(merged_dataset)

    def merge_with_representatives_2d(self, spatial_clusters, representatives):
        '''
        Similarly to merge_clusters_2d, returns a spatial region. However, the value at each point
        is the value from the specified representative of a cluster, to which the point belongs.
        '''
        # sanity check: as many representatives as there are clusters
        assert len(spatial_clusters) == len(representatives)

        # prepare new clusters that have the value at the representatives copied.
        # this will fail if the representative is not a member of its cluster!
        repeated_clusters = []
        for cluster_index, cluster in enumerate(spatial_clusters):
            repeated_cluster = cluster.repeat_point(representatives[cluster_index])
            repeated_clusters.append(repeated_cluster)

        # now call merge, should work as expected: the values in each cluster are the same value
        # as their own representative
        return self.merge_clusters_2d(repeated_clusters)

    def find_indices_of_clusters_intersecting_with(self, region_2d):
        '''
        Given a 2d region, find the clusters that share common points (i.e. intersect) with it.
        Returns a list of integers representing the cluster indices.

        Assumes that the coordinates of this partition match with the region (same origin).
        '''
        # find the membership of each point in the region using its index
        # the index needs to be calculated for each point
        point_indices = []
        for x in range(region_2d.x1, region_2d.x2):
            for y in range(region_2d.y1, region_2d.y2):
                point_index_x_y = x * self.y_len + y
                point_indices.append(point_index_x_y)
        point_memberships = self.membership_of_point_indices(point_indices)

        # this retrieves indices without repetition
        return list(set(point_memberships))

    def find_medoids_of_clusters_intersecting_with(self, region_2d):
        '''
        Given a 2d region, find the medoids (as points) of the clusters that share common points
        (i.e. intersect) with it.
        Uses find_indices_of_clusters_intersecting_with(region_2d) to find the indices.
        '''
        # need the medoids...
        if not hasattr(self, 'medoids'):
            raise ValueError('Partition does not have the required medoids!')

        # first find the intersected clusters, then retrieve their medoids using the index
        # assuming that the medoids are ordered by the cluster index!
        point_memberships = self.find_indices_of_clusters_intersecting_with(region_2d)
        return [self.medoids[cluster_index] for cluster_index in point_memberships]

    @classmethod
    def from_membership_array(cls, membership, x_len, y_len):
        '''
        Creates an instance of PartitionRegionCrisp using a 1-d membership array as input.
        Requires the region shape.
        '''
        # find k: the maximum index in the array + 1 (values range from 0 to k-1)
        k = int(np.max(membership) + 1)

        # reshape the array to get a membership matrix (shape of region)
        membership_2d = membership.reshape(x_len, y_len)

        return PartitionRegionCrisp(membership_2d, k)

    @classmethod
    def with_regular_partition(cls, k, x_len, y_len):
        '''
        Creates an instance of PartitionRegionCrisp based on the number of clusters, so that each
        cluster gets a rectangle of (approximately) the same size.
        The process is as follows:
            - Compute the most balanced divisors of k, e.g. 12 -> 4 x 3.
            - Partition the (x_len, y_len)  using these two values to create the cluster labels,
              e.g.
                  0  0  0  1  1  1 ...  3  3  3
                  0  0  0  1  1  1 ...  3  3  3
                  .....................
                  9  9  9 10 10 10 ..  11 11 11
                  9  9  9 10 10 10 ..  11 11 11

            - call from _membership_array with this membership matrix
        '''
        # all clusters in the partition will have this same membership matrix
        membership_regular = arrays_util.regular_partitioning(x_len, y_len, k)
        return cls.from_membership_array(membership_regular, x_len, y_len)


class PartitionRegionFuzzy(PartitionRegion):

    # TODO
    # use this for threshold? https://stackoverflow.com/a/38532088/3175179
    pass


def intra_cluster_cost(partition, spt_region, distance_measure):
    '''
    Given a cluster partition, calculate the total intra-cluster cost.

    This is calculated using the SSE (sum of squared errors), where the "error" is the distance of each member
    of the partition to its corresponding medoid.

    Assumes that the medoids are available in the partition.
    '''
    sum_of_intra_cluster_cost = 0

    medoids = partition.medoids

    # work with clusters
    clusters = partition.create_all_spt_clusters(spt_region, medoids=medoids)

    for cluster_index, cluster in enumerate(clusters):

        cluster_medoid = medoids[cluster_index]

        # the distance measure requires point indices
        cluster_points = [point for (point, _) in cluster]
        cluster_point_indices = [
            point.x * spt_region.y_len + point.y
            for point in cluster_points
        ]

        # use the distance_measure to calculate all distances
        distances_to_medoid = distance_measure.distances_to_point(spt_region, cluster_medoid, cluster_point_indices)
        # sum_of_intra_cluster_cost += np.sum(intra_cluster_cost)
        sum_of_intra_cluster_cost += arrays_util.sum_squared(distances_to_medoid)

    return sum_of_intra_cluster_cost


if __name__ == '__main__':
    logger = log_util.setup_log('DEBUG')

    from spta.clustering.kmedoids import KmedoidsClusteringMetadata, KmedoidsClusteringAlgorithm
    from spta.distance.dtw import DistanceByDTW

    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata

    region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                   2015, 2015, 1, scaled=False)
    spt_region = region_metadata.create_instance()
    distance_measure = DistanceByDTW()

    kmedoids_metadata = KmedoidsClusteringMetadata(k=4, random_seed=1)
    kmedoids_algorithm = KmedoidsClusteringAlgorithm(kmedoids_metadata, distance_measure)
    partition = kmedoids_algorithm.partition(spt_region, pickle_home='pickle')
    logger.debug('Got partition: {} with medoids {}'.format(partition, partition.medoids))

    cost = intra_cluster_cost(partition, spt_region, distance_measure)
    logger.debug('Cost for {} -> {}'.format(kmedoids_metadata, cost))
