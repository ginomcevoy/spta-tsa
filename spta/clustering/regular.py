from collections import OrderedDict

from spta.region.partition import PartitionRegionCrisp

from . import ClusteringMetadata, ClusteringAlgorithm


class RegularClusteringMetadata(ClusteringMetadata):
    '''
    Stores metadata for a clustering algorithm.
    '''

    def __init__(self, k):
        super(RegularClusteringMetadata, self).__init__('regular', k)

    def as_dict(self):
        return OrderedDict([
            ('type', 'regular'),
            ('k', self.k),
        ])

    @classmethod
    def from_repr(cls, repr_string):
        '''
        Given the string representation, recreate the instance.
        '''
        parts = repr_string.split('_')
        assert(parts[0] == 'regular')

        # only k, e.g. k2
        k_string = parts[1]
        k = int(k_string[1:])

        return RegularClusteringMetadata(k)


def regular_metadata_generator(k_values):
    '''
    Generate metadata given a list of k_values
    '''
    # FIXME no identifier here yet, so the caller MUST set it manually afterwards.
    # TODO refactor this method so that the identifier is passed.
    # Right now we don't want to change experiments.metadata...
    # NOTE: importing here to break import cycle (factory -> kmedoids -> suite -> factory)
    from . import suite
    return suite.ClusteringSuite('change_me', 'regular', k=k_values)


class RegularClusteringAlgorithm(ClusteringAlgorithm):

    def partition_2d(self, spatial_region):
        '''
        Create a regular partition for a spatial region, no medoids.
        '''
        # The partition method also works for spatial regions, notice how the shape is extracted.
        return self.partition(spatial_region, with_medoids=False)

    def partition_impl(self, spt_region, with_medoids=True, save_csv_at=None):
        '''
        Create a regular partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters.

        with_medoids
            Optionally return the medoids of the clusters
        '''
        self.logger.info('Clustering algorithm {}'.format(self.metadata))

        x_len, y_len = spt_region.x_len, spt_region.y_len

        partition = PartitionRegionCrisp.with_regular_partition(self.k, x_len, y_len)

        if with_medoids:
            medoids = self.find_medoids_for_partition(spt_region, partition)

            # save medoids as member of the partition!
            partition.medoids = medoids

        return partition
