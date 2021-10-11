import numpy as np
from sklearn.cluster import DBSCAN
# from sklearn import metrics
from collections import OrderedDict

from spta.region.partition import PartitionRegionCrisp

from . import ClusteringMetadata, ClusteringAlgorithm


class DbscanClusteringMetadata(ClusteringMetadata):
    '''
    Stores metadata for a clustering algorithm.
    '''

    def __init__(self, k=0):
        # Dbscan does not store a k value by default: it requires no value for k when running,
        # but it can return a value for k after its execution.
        super(DbscanClusteringMetadata, self).__init__('dbscan', k)

    def as_dict(self):
        return OrderedDict([
            ('type', 'dbscan'),
            ('k', self.k),
        ])

    @classmethod
    def from_repr(cls, repr_string):
        '''
        Given the string representation, recreate the instance.
        '''
        parts = repr_string.split('_')
        assert(parts[0] == 'dbscan')

        # only k, e.g. k2
        k_string = parts[1]
        k = int(k_string[1:])

        return DbscanClusteringMetadata(k)


def dbscan_metadata_generator(k_values):
    '''
    Generate metadata given a list of k_values
    '''
    # FIXME no identifier here yet, so the caller MUST set it manually afterwards.
    # TODO refactor this method so that the identifier is passed.
    # Right now we don't want to change experiments.metadata...
    # NOTE: importing here to break import cycle (factory -> kmedoids -> suite -> factory)
    from . import suite
    return suite.ClusteringSuite('change_me', 'dbscan', k=k_values)


class DbscanClusteringAlgorithm(ClusteringAlgorithm):

    def partition_2d(self, spatial_region):
        '''
        Create a dbscan partition for a spatial region, no medoids.
        '''
        # The partition method also works for spatial regions, notice how the shape is extracted.
        return self.partition(spatial_region, with_medoids=False)

    def partition_impl(self, spt_region, with_medoids=True, save_csv_at=None):
        '''
        Create a dbscan partition on a spatio-temporal region. A partition can be used to create
        spatio-temporal clusters.

        with_medoids
            Optionally return the medoids of the clusters
        '''
        self.logger.info('Clustering algorithm {}'.format(self.metadata))

        db = DBSCAN(eps=0.7, min_samples=5, metric='precomputed',
                    algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(distance_dtw.distance_matrix)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_ = len(set(labels))
        n_noise_ = list(labels).count(-1)

        # consider the noise points as a separate group with cluster_index 0
        labels_with_noise = labels + 1
        print(labels_with_noise)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        # print("Adjusted Rand Index: %0.3f"
        #       % metrics.adjusted_rand_score(labels_true, labels))
        # print("Adjusted Mutual Information: %0.3f"
        #       % metrics.adjusted_mutual_info_score(labels_true, labels))
        # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(spt_region.as_2d, labels))

        _, x_len, y_len = spt_region.shape
        partition = PartitionRegionCrisp.from_membership_array(labels_with_noise,
                                                               x_len, y_len)
        return partition


if __name__ == '__main__':

    from spta.dataset.metadata import TemporalMetadata, SamplesPerDay
    from spta.distance.dtw import DistanceByDTW, DistanceBySpatialDTW
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.util import log as log_util
    from spta.util import plot as plot_util

    log_util.setup_log('DEBUG')

    # Run k=2 on sp_small dataset
    # region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
    #                                                2015, 2015, 1)
    # region_metadata = SpatioTemporalRegionMetadata('whole_brazil', Region(20, 100, 15, 95),
    #                                                2015, 2015, 1, scaled=True)

    dataset_class_name = 'spta.dataset.csfr.DatasetCSFR'
    temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
    region_metadata = SpatioTemporalRegionMetadata(name='whole_brazil',
                                                   region=Region(20, 100, 15, 95),
                                                   temporal_md=temporal_md,
                                                   dataset_class_name=dataset_class_name,
                                                   scaled=False)

    spt_region = region_metadata.create_instance()

    # load pre-computed distances
    distance_dtw = DistanceByDTW()
    distance_dtw = DistanceBySpatialDTW(0.2)
    distance_dtw.load_distance_matrix_2d(region_metadata.distances_filename,
                                         region_metadata.region)

    dbscan_metadata = DbscanClusteringMetadata()
    dbscan_algorithm = DbscanClusteringAlgorithm(dbscan_metadata, distance_dtw)
    partition = dbscan_algorithm.partition(spt_region)

    plot_util.plot_partition(partition, title='DBSCAN partition')
