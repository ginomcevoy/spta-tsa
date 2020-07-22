from spta.region import Point
from spta.region.partition import PartitionRegionCrisp
from spta.kmedoids.kmedoids import run_kmedoids

from . import ClusteringMetadata, ClusteringAlgorithm


class KmedoidsClusteringMetadata(ClusteringMetadata):
    '''
    Stores metadata for k-medoids clustering algorithm.
    '''

    def __init__(self, k, random_seed=1, initial_medoids=None, max_iter=1000, tol=0.001,
                 verbose=True):
        '''
        Sets up the k-medoids metadata with default values. The value of k is still required.
        '''
        super(KmedoidsClusteringMetadata, self).__init__('kmedoids', k=k)
        self.random_seed = random_seed
        self.initial_medoids = initial_medoids
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def as_dict(self):
        return {
            'type': 'kmedoids',
            'k': self.k,
            'seed': self.random_seed
        }

    def __repr__(self):
        r = '{}_k{}_seed{}'.format(self.name, self.k, self.random_seed)
        if self.initial_medoids is not None:
            indices_str = [
                str(index)
                for index
                in self.initial_medoids
            ]
            indices_str = '-'.join(indices_str)
            r = '{}_im{}'.format(r, indices_str)

        return r

    def __str__(self):
        '''
        Useful for plot titles
        '''
        return '{}: k={} seed={}'.format(self.name.capitalize(), self.k, self.random_seed)


def kmedoids_metadata_generator(k_values, seed_values, initial_medoids=None, max_iter=1000,
                                tol=0.001, verbose=True):
    '''
    Generate k-medoids metadata given a list of k_values and seeds, also default values for
    the other parameters. Performs a cartesian product of k_values and seeds.
    '''
    for k in k_values:
        for random_seed in seed_values:
            yield KmedoidsClusteringMetadata(k, random_seed, initial_medoids, max_iter, tol,
                                             verbose)


class KmedoidsClusteringAlgorithm(ClusteringAlgorithm):

    def partition(self, spt_region, with_medoids=True, save_csv_at=None):
        '''
        Create a k-medoids partition on a spatio-temporal region.
        A partition can be used to create spatio-temporal clusters.

        with_medoids
            Optionally return the medoids of the clusters
        '''
        self.logger.info('Clustering algorithm {}'.format(self.metadata))

        # the k-medoids algorithm works on list of series, not on spatio-temporal regions
        X = spt_region.as_2d
        _, x_len, y_len = spt_region.shape

        # run k-medoids algorithm
        kmedoids_result = run_kmedoids(X, self.k, self.distance_measure,
                                       initial_medoids=self.metadata.initial_medoids,
                                       random_seed=self.metadata.random_seed,
                                       max_iter=self.metadata.max_iter,
                                       tol=self.metadata.tol,
                                       verbose=self.metadata.verbose)

        # build result
        partition = PartitionRegionCrisp.from_membership_array(kmedoids_result.labels,
                                                               x_len, y_len)

        # save CSV?
        if save_csv_at is not None:
            self.save_to_csv(partition, spt_region.region_metadata, save_csv_at)

        if with_medoids:

            # run_kmedoids returns indices, not Point instances
            medoids = [
                Point(int(centroid_index / y_len), centroid_index % y_len)
                for centroid_index
                in kmedoids_result.medoids
            ]

            return partition, medoids

        else:
            # no medoids
            return partition
