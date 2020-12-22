import matplotlib.pyplot as plt
import os

from .factory import ClusteringFactory

from spta.util import log as log_util
from spta.util import plot as plot_util


class SilhouetteAnalysis(log_util.LoggerMixin):

    def __init__(self, region_metadata, distance_measure, clustering_suite):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.clustering_suite = clustering_suite

        self.spt_region = region_metadata.create_instance()
        self.clustering_factory = ClusteringFactory(distance_measure)

    def perform_analysis(self, output_home, pickle_home):
        '''
        Here the silhouette analysis is performed: for every clustering algorithm (k/seed combination),
        the labels are used to calculate a silhouette score. The best (highest) score is saved and the
        corresponding clustering algorithm is returned.
        '''

        # load pre-computed distances
        if self.distance_measure.distance_matrix is None:
            self.distance_measure.load_distance_matrix_2d(self.region_metadata.distances_filename,
                                                          self.region_metadata.region)

        # updated at every iteration
        best_silhouette_avg = -1
        best_clustering_algorithm = None

        for clustering_metadata in self.clustering_suite:

            clustering_algorithm = self.clustering_factory.instance(clustering_metadata)
            self.logger.debug('Analyzing silhouette for {}'.format(clustering_metadata))

            # use the clustering algorithm to get the partition and medoids
            # will try to leverage pickle and load previous attempts, otherwise calculate and save
            partition = clustering_algorithm.partition(self.spt_region, with_medoids=True,
                                                       save_csv_at=output_home,
                                                       pickle_home=pickle_home)

            # the silhouette for the current clustering algorithm
            silhouette_avg = self.single_silhouette(partition, clustering_algorithm, output_home)
            self.logger.debug('silhouette_avg -> {}'.format(silhouette_avg))

            # save best results
            if silhouette_avg > best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_clustering_algorithm = clustering_algorithm

        self.logger.info('best_silhouette_avg {} -> {}'.format(best_silhouette_avg, best_clustering_algorithm))
        return (best_silhouette_avg, best_clustering_algorithm)

    def single_silhouette(self, partition, clustering_algorithm, output_home):
        '''
        Here the silhouette is calculated using a helper function at plot_util. The parameters are adapted
        for the lower-level function.

        This function will always save and show all the graphs.
        '''
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        title = 'Silhouette analysis for {}'.format(clustering_algorithm)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        self.logger.debug(title)

        # building the silhouette graph requires all distances
        distance_matrix = self.distance_measure.distance_matrix
        assert distance_matrix is not None

        # the low-level function requires 1-D labels, we can extract them from the partition
        _, x_len, y_len = self.spt_region.shape
        labels = partition.numpy_dataset.reshape(x_len * y_len)
        shape_2d = (x_len, y_len)

        # plot the clustering in 2d
        plot_util.plot_2d_clusters(labels, shape_2d, title='Clustering', subplot=ax1)

        # this computes the silhouette average *and* creates the plot
        silhouette_avg = plot_util.plot_clustering_silhouette(distance_matrix, labels, subplot=ax2)

        # always show and save graph...
        plt.show()

        output_dir = clustering_algorithm.metadata.output_dir(output_home, self.region_metadata, self.distance_measure)
        silhouette_filename = 'silhouette-{!r}.pdf'.format(clustering_algorithm)
        silhouette_filepath = os.path.join(output_dir, silhouette_filename)
        self.logger.debug('Saving silhouette at {}'.format(silhouette_filepath))

        fig.savefig(silhouette_filepath)

        return silhouette_avg


if __name__ == '__main__':

    from spta.clustering.kmedoids import kmedoids_metadata_generator
    from spta.distance.dtw import DistanceByDTW
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.util import log as log_util

    log_util.setup_log('DEBUG')

    region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                   2015, 2015, 1, scaled=False)
    clustering_suite = kmedoids_metadata_generator(k_values=range(2, 4), seed_values=range(0, 2))
    clustering_suite.identifier = 'quick'

    distance_dtw = DistanceByDTW()

    silhouette_analysis = SilhouetteAnalysis(region_metadata, distance_dtw, clustering_suite)
    silhouette_analysis.perform_analysis('outputs', 'pickle')
