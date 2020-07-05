import os

from spta.region.centroid import CalculateCentroid
from spta.region.partition import PartitionRegion

from spta.util import log as log_util


class ClusteringMetadata():
    '''
    Stores metadata for a clustering algorithm. The common parameters are:

        name
            identifies the algorithm
        k
            the number of clusters
    '''

    def __init__(self, name, k, **kwargs):
        self.name = name
        self.k = k

    def clustering_subdir(self, region_metadata, distance_measure):
        '''
        Sub-directory used for saving results.
        '''
        region_subdir = '{!r}'.format(region_metadata)
        clustering_metadata_subdir = '{!r}'.format(self)
        distance_subdir = '{!r}'.format(distance_measure)
        return os.path.join(region_subdir, clustering_metadata_subdir, distance_subdir)

    def csv_dir(self, region_metadata, distance_measure):
        '''
        Directory to store CSV results.
        '''
        return os.path.join('csv', self.clustering_subdir(region_metadata, distance_measure))

    def pickle_dir(self, region_metadata, distance_measure):
        '''
        Directory to store pickle objects.
        '''
        return os.path.join('pickle', self.clustering_subdir(region_metadata, distance_measure))

    def plot_dir(self, region_metadata, distance_measure):
        '''
        Directory to store plots results.
        '''
        return os.path.join('plot', self.clustering_subdir(region_metadata, distance_measure))

    def __repr__(self):
        return '{}_k{}'.format(self.name, self.k)

    def __str__(self):
        return '{}: k={}'.format(self.name.capitalize(), self.k)


class ClusteringAlgorithm(log_util.LoggerMixin):
    '''
    A clustering algorithm that will produce a partition when applied to a region.
    '''

    def __init__(self, metadata, distance_measure):
        self.metadata = metadata
        self.distance_measure = distance_measure
        self.k = metadata.k

    def partition(self, spt_region, with_medoids=True):
        '''
        Create a partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters.

        with_medoids
            Optionally return the medoids of the clusters
        '''
        raise NotImplementedError

    def find_medoids_for_partition(self, spt_region, partition):
        '''
        Helper method to find the medoids given the partition. This is required when using the
        "regular" clustering algorithm, which does not calculate the medoids.

        partition
            an instance of spta.region.partition.PartitionRegion
        '''
        assert isinstance(partition, PartitionRegion)

        # algorithm to find the medoid of a cluster given a distance measure
        calculate_centroid = CalculateCentroid(self.distance_measure)

        # create the clusters without medoids
        clusters = partition.create_all_spt_clusters(spt_region, centroid_indices=None,
                                                     medoids=None)

        # find all medoids using the medoid algorithm
        medoids = [
            calculate_centroid.find_centroid_and_distances(cluster)[0]
            for cluster
            in clusters
        ]

        return medoids
