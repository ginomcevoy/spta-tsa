from collections import OrderedDict

from spta.region import Point
from spta.region.partition import PartitionRegionCrisp
from spta.kmedoids.kmedoids import run_kmedoids

from . import ClusteringMetadata, ClusteringAlgorithm


class KmedoidsClusteringMetadata(ClusteringMetadata):
    '''
    Stores metadata for k-medoids clustering algorithm.
    '''

    def __init__(self, k, random_seed=1, mode='lite', initial_medoids=None, max_iter=1000,
                 tol=0.001, verbose=True):
        '''
        Sets up the k-medoids metadata with default values. The value of k is still required.
        '''
        super(KmedoidsClusteringMetadata, self).__init__('kmedoids', k=k)
        self.random_seed = random_seed
        self.initial_medoids = initial_medoids
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def as_dict(self):
        return OrderedDict([
            ('type', 'kmedoids'),
            ('k', self.k),
            ('seed', self.random_seed),
            ('mode', self.mode)
        ])

    def __repr__(self):
        r = '{}_k{}_seed{}_{}'.format(self.name, self.k, self.random_seed, self.mode)
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
        return '{}: k={} seed={} mode={}'.format(self.name.capitalize(), self.k, self.random_seed,
                                                 self.mode)

    @classmethod
    def from_repr(cls, repr_string):
        '''
        Given the representation, recover the instance.
        NOTE: This implementation assumes that initial_medoids has not been set!
        '''
        parts = repr_string.split('_')
        assert(parts[0] == 'kmedoids')

        # 1. k
        k_string = parts[1]
        k = int(k_string[1:])  # e.g. k2

        # 2. seed
        seed_string = parts[2]
        random_seed = int(seed_string[4:])   # e.g. seed5

        # 3. mode
        mode = parts[3]   # e.g. lite
        return KmedoidsClusteringMetadata(k, random_seed=random_seed, mode=mode)


def kmedoids_metadata_generator(k_values, seed_values, mode='lite', initial_medoids=None,
                                max_iter=1000, tol=0.001, verbose=True):
    '''
    Generate k-medoids metadata given a list of k_values and seeds, also default values for
    the other parameters. Performs a cartesian product of k_values and seeds.
    '''
    # FIXME no identifier here yet, so the caller MUST set it manually afterwards.
    # TODO refactor this method so that the identifier is passed.
    # Right now we don't want to change experiments.metadata...
    # NOTE: importing here to break import cycle (factory -> kmedoids -> suite -> factory)
    from . import suite
    return suite.ClusteringSuite('change_me', 'kmedoids', k=k_values, random_seed=seed_values,
                                 mode=mode, initial_medoids=initial_medoids, max_iter=max_iter,
                                 tol=tol, verbose=verbose)


class KmedoidsClusteringAlgorithm(ClusteringAlgorithm):

    def partition_impl(self, spt_region, with_medoids=True):
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
                                       mode=self.metadata.mode,
                                       max_iter=self.metadata.max_iter,
                                       tol=self.metadata.tol,
                                       verbose=self.metadata.verbose)

        # build result
        partition = PartitionRegionCrisp.from_membership_array(kmedoids_result.labels,
                                                               x_len, y_len)

        if with_medoids:

            # run_kmedoids returns indices, not Point instances
            medoids = [
                Point(int(centroid_index / y_len), centroid_index % y_len)
                for centroid_index
                in kmedoids_result.medoids
            ]

            # save medoids as member of the partition!
            partition.medoids = medoids

        return partition
