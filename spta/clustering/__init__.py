import csv
import os

from spta.region.centroid import CalculateCentroid
from spta.region.partition import PartitionRegion

from spta.util import log as log_util
from spta.util import fs as fs_util


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
        <region>/<distance>/<clustering>
        '''
        region_subdir = '{!r}'.format(region_metadata)
        distance_subdir = '{!r}'.format(distance_measure)
        clustering_metadata_subdir = '{!r}'.format(self)
        return os.path.join(region_subdir, distance_subdir, clustering_metadata_subdir)

    def output_dir(self, output_prefix, region_metadata, distance_measure):
        '''
        Directory to store CSV and plots.
        <output_prefix>/<region>/<distance>/<clustering>
        '''
        return os.path.join(output_prefix, self.clustering_subdir(region_metadata,
                                                                  distance_measure))

    def pickle_dir(self, region_metadata, distance_measure):
        '''
        Directory to store pickle objects.
        pickle/<region>/<distance>/<clustering>
        '''
        return os.path.join('pickle', self.clustering_subdir(region_metadata, distance_measure))

    def as_dict(self):
        '''
        Returns a representation of this metadata as a list of elements
        '''
        pass

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

    def partition(self, spt_region, with_medoids=True, save_csv_at=None):
        '''
        Create a partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters.

        with_medoids
            Optionally return the medoids of the clusters

        save_csv_at
            Optionally save a CSV report, at the specified path
            TODO this is ugly, improve?
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

    def output_dir(self, output_prefix, region_metadata):
        '''
        Directory for CSV files and plots, delegates to clustering metadata
        '''
        return self.metadata.output_dir(output_prefix=output_prefix,
                                        region_metadata=region_metadata,
                                        distance_measure=self.distance_measure)

    def save_to_csv(self, partition, region_metadata, output_prefix):
        '''
        Create a CSV report of the clustering partition.
        Requires the region metadata to store the CSV in a proper path.

        cluster points coverage(#points/#total points)
        '''

        # path to store CSV
        csv_output_dir = self.output_dir(output_prefix, region_metadata)
        fs_util.mkdir(csv_output_dir)

        csv_filename = 'clustering__{!r}.csv'.format(self)
        csv_filepath = os.path.join(csv_output_dir, csv_filename)

        with open(csv_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            # header
            csv_writer.writerow(['cluster', 'points', 'coverage'])

            # total number of points in region, need to find the coverage
            # abusing the partition a bit to get this
            total_points = partition.shape[0] * partition.shape[1]

            # iterate clusters in the partition, no need to create full-blown cluster objects
            for i in range(0, self.k):

                # points is the number of points in the cluster
                points_i = partition.cluster_len(i)

                # coverage is the % of points in the cluster, relative to total number of points
                coverage_i = points_i * 100.0 / total_points
                coverage_i_str = '{:.1f}'.format(coverage_i)

                csv_writer.writerow([str(i), str(points_i), coverage_i_str])

        self.logger.info('Saved {} CSV at: {}'.format(self, csv_filepath))

    def as_dict(self):
        return self.metadata.as_dict()

    def __repr__(self):
        return repr(self.metadata)

    def __str__(self):
        return str(self.metadata)
