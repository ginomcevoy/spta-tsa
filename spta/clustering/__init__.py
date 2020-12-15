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
        self.k = int(k)

    def clustering_subdir(self, distance_measure):
        '''
        Sub-directory used for saving results, relative to the region dir.
        <region>/<distance>/<clustering>
        '''
        distance_subdir = '{!r}'.format(distance_measure)
        clustering_metadata_subdir = '{!r}'.format(self)
        return os.path.join(distance_subdir, clustering_metadata_subdir)

    def output_dir(self, output_home, region_metadata, distance_measure):
        '''
        Directory to store CSV and plots, e.g.
        <output_home>/<region>/<distance>/<clustering>
        '''
        region_output_dir = region_metadata.output_dir(output_home)
        return os.path.join(region_output_dir, self.clustering_subdir(distance_measure))

    def pickle_dir(self, region_metadata, distance_measure, pickle_home='pickle'):
        '''
        Directory to store pickle objects, e.g.
        pickle/<region>/<distance>/<clustering>
        '''
        region_pickle_dir = region_metadata.pickle_dir(pickle_home)
        return os.path.join(region_pickle_dir, self.clustering_subdir(distance_measure))

    def as_dict(self):
        '''
        Returns a representation of this metadata as a list of elements
        '''
        pass

    def __repr__(self):
        return '{}_k{}'.format(self.name, self.k)

    def __str__(self):
        return '{}: k={}'.format(self.name.capitalize(), self.k)

    def __eq__(self, other):
        return self.name == other.name and self.k == other.k

    def __hash__(self):
        return hash((self.name, self.k))


class ClusteringAlgorithm(log_util.LoggerMixin):
    '''
    A clustering algorithm that will produce a partition when applied to a region.
    Subclasses must implement the method partition_impl(spt_region, with_medoids)
    '''

    def __init__(self, metadata, distance_measure):
        self.metadata = metadata
        self.distance_measure = distance_measure
        self.k = metadata.k

        # filename to store partitions
        self.partition_pickle_filename = 'partition_{!r}.pkl'.format(self.metadata)

    def partition(self, spt_region, with_medoids=True, save_csv_at=None, pickle_home=None):
        '''
        Create a partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters. Calls subclass implementation of partition_impl

        with_medoids
            Optionally return the medoids of the clusters

        save_csv_at
            Optionally save a CSV report, at the specified prefix
            TODO this is ugly, improve?

        pickle_home
            Optionally save the partition as a pickle object at the specified prefix
        '''

        # if we can load the partition, then use saved result
        # can only be done if region metadata is available
        loaded_from_pickle = False
        if pickle_home is not None and spt_region.region_metadata is not None:
            the_partition = self.try_load_previous_partition(spt_region.region_metadata,
                                                             pickle_home)

            # flag to avoid saving a loaded partition
            if the_partition is not None:
                loaded_from_pickle = True

        if not loaded_from_pickle:

            # could not find previous partition, call the logic of a subclass to find the partition
            # but first try to load the distance matrix if it is available
            self.distance_measure.try_load_distance_matrix(spt_region)
            the_partition = self.partition_impl(spt_region, with_medoids)

        if save_csv_at is not None:
            # creates a CSV report of this clustering, assumes region metadata is available
            self.save_to_csv(the_partition, spt_region.region_metadata, save_csv_at)

        if pickle_home is not None and not loaded_from_pickle:
            # saves the partition as a pickle object for later retrieval, assumes region metadata
            # is available
            self.save_partition(the_partition, spt_region.region_metadata, pickle_home)

        return the_partition

    def partition_impl(self, spt_region, with_medoids):
        '''
        Create a partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters. Subclass must implementat this.
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

    def output_dir(self, output_home, region_metadata):
        '''
        Directory for CSV files and plots, delegates to clustering metadata
        '''
        return self.metadata.output_dir(output_home=output_home,
                                        region_metadata=region_metadata,
                                        distance_measure=self.distance_measure)

    def save_to_csv(self, partition, region_metadata, output_home):
        '''
        Create a CSV report of the clustering partition.
        Requires the region metadata to store the CSV in a proper path.

        cluster points coverage(#points/#total points)
        '''

        # path to store CSV
        csv_output_dir = self.output_dir(output_home, region_metadata)
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

            # iterate indices in the partition, no need to create full-blown cluster objects
            for i in range(0, self.k):

                # points is the number of points in the cluster
                points_i = partition.cluster_len(i)

                # coverage is the % of points in the cluster, relative to total number of points
                coverage_i = points_i * 100.0 / total_points
                coverage_i_str = '{:.1f}'.format(coverage_i)

                csv_writer.writerow([str(i), str(points_i), coverage_i_str])

        self.logger.info('Saved {} CSV at: {}'.format(self, csv_filepath))

    def save_partition(self, partition, region_metadata, pickle_home='pickle'):
        '''
        Saves a partition as a pickle objct, using its to_pickle(path) method. The path is
        calculated by the clustering metadata.
        '''
        pickle_full_path = self.pickle_full_path(region_metadata)

        # the partition can save itself
        partition.to_pickle(pickle_full_path)

    def try_load_previous_partition(self, region_metadata, pickle_home='pickle'):
        '''
        Tries to load a partition that has been previously calculated with this clustering
        algorithm for the specified region metadata, and returns the partition instance.

        Should be successful if save_partition() was called before.
        If no saved partition is found, returns None without raising errors.
        '''
        # try and load the partition object, this call may fail
        partition = None
        try:
            partition = self.load_previous_partition(region_metadata, pickle_home)
        except Exception:
            # attempt failed, return None without exception
            self.logger.debug('Attempt to load a partition with {!r} failed.'.format(self))

        return partition

    def load_previous_partition(self, region_metadata, pickle_home='pickle'):
        '''
        Load a partition that has been previously calculated with this clustering
        algorithm for the specified region metadata, and returns the partition instance.

        Should be successful if save_partition() was called before, will raise an exception
        if no saved partition is found.
        '''
        pickle_full_path = self.pickle_full_path(region_metadata)
        partition = PartitionRegion.try_from_pickle(pickle_full_path)

        self.logger.info('Loaded previously calculated partition from {}'.format(pickle_full_path))
        return partition

    def pickle_full_path(self, region_metadata):
        '''
        Calculates the path for persisting partition objects via pickle. The path is calculated by
        the clustering metadata.
        '''
        # path to store/load pickle objects given metadata
        pickle_dir = self.metadata.pickle_dir(region_metadata, self.distance_measure)

        # use a fixed name, assuming that the metadata uniquely identifies the created partition...
        return os.path.join(pickle_dir, self.partition_pickle_filename)

    def as_dict(self):
        return self.metadata.as_dict()

    def __repr__(self):
        return repr(self.metadata)

    def __str__(self):
        return str(self.metadata)
