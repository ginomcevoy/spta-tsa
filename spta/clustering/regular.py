from spta.region.partition import PartitionRegionCrisp

from . import ClusteringMetadata, ClusteringAlgorithm


class RegularClusteringMetadata(ClusteringMetadata):
    '''
    Stores metadata for a clustering algorithm.
    '''

    def __init__(self, k):
        super(RegularClusteringMetadata, self).__init__('regular', k)

    def as_dict(self):
        return {
            'type': 'regular',
            'k': self.k,
        }


def regular_metadata_generator(k_values):
    '''
    Generate metadata given a list of k_values
    '''
    for k in k_values:
        yield RegularClusteringMetadata(k)


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
