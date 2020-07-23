import logging
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import os
import sys

from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.util import plot as plot_util
from spta.distance.dtw import DistanceByDTW

from . import kmedoids

# logging: avoid showing library outputs
logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.backends.backend_ps').disabled = True

''' The metadata for a silhouette analysis '''
SilhouetteMetadata = namedtuple('SilhouetteMetadata',
                                ('k_values', 'seed_values', 'kmedoids_mode', 'distance_measure',
                                 'initial_medoids', 'max_iter', 'tol', 'verbose', 'show_graphs',
                                 'save_graphs'))


class KmedoidsWithSilhouette(object):
    '''
    Performs K-medoids and Silhouette analysis over a spatio temporal region.
    The region is specified by its metadata, the correct dataset will be saved (save command).
    The distance matrix can be calculated (distance command), and then the k-medoids will be
    performed to find the centroids and the display the graphs (kmedoids command)

    TODO: better to separate these commands?
    '''

    def __init__(self, sptr_metadata, silhouette_metadata):
        self.sptr_metadata = sptr_metadata
        self.silhouette_metadata = silhouette_metadata

        # the commands accepted by this analysis
        self.commands = {
            'save': self.save_sptr,
            'distances': self.calculate_distance_matrix,
            'kmedoids': self.silhouette_analysis,
            'show_distances': self.show_distances
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def save_sptr(self):
        '''
        Obtains the spatio temporal region from a complete dataset, and saves it to a file.
        '''
        # read the dataset to load the spatio-temporal region
        sptr = self.sptr_metadata.create_instance()

        # save it to file given by metadata
        sptr.save()

    def calculate_distance_matrix(self):
        '''
        Calculates the distance matrix using a DistanceMeasure instance and saves it to a file.
        This can take a long time, it is O((x_len * y_len)^2).
        '''
        # load data of spatio-temporal region from file
        sptr = self.__load_saved_spt_region()

        self.logger.info('Calculating distances...')
        distance_measure = self.silhouette_metadata.distance_measure
        distance_matrix = distance_measure.compute_distance_matrix(sptr)
        np.save(self.sptr_metadata.distances_filename, distance_matrix)

    def silhouette_analysis(self):
        '''
        Performs the silhouette analysis given the provided metadata.
        '''
        # load data of spatio-temporal region from file
        spt_region = self.__load_saved_spt_region()
        shape_2d = spt_region.shape_2d

        # use pre-computed distance matrix
        distance_measure = self.silhouette_metadata.distance_measure
        distance_measure.load_distance_matrix_2d(self.sptr_metadata.distances_filename,
                                                 self.sptr_metadata.region)

        # perform the analysis
        # this will iterate over given seeds and k values.
        silhouette_result = do_silhouette_analysis(spt_region, self.silhouette_metadata)
        (best_k, best_seed, best_medoids, best_labels) = silhouette_result

        # Show best results
        logger.info('Best k: {}'.format(best_k))
        logger.info('Best seed: {}'.format(best_seed))
        logger.info('Best medoids: {}'.format(str(best_medoids)))
        logger.info('Best labels: {}'.format(str(best_labels)))

        # print results for best k
        # Create a subplot with 1 row and 2 columns
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        plt.suptitle('Best silhouette score, k={}'.format(best_k), fontsize=14, fontweight='bold')

        # the clustering
        plot_util.plot_2d_clusters(best_labels, shape_2d, title='Clustering', subplot=ax1)

        # build the silhouette graph, requires all distances
        plot_util.plot_clustering_silhouette(distance_measure.distance_matrix,
                                             best_labels, subplot=ax2)
        plt.show()

    def show_distances(self):
        '''
        Show 2d graphs for the distances to the first and center points.
        Uses the distance matrix to get the distances to point (0, 0) and to the point that
        is at the center of the graph.

        Default filenames for output files:
        plot/distances_0_0_<name>.eps
        plot/distances_center_<name>.eps

        where <name> is the name in the region metadata.
        '''
        # load data of spatio-temporal region from file
        spt_region = self.__load_saved_spt_region()
        (x_len, y_len) = spt_region.shape_2d

        # use pre-computed distance matrix
        distance_measure = self.silhouette_metadata.distance_measure
        distance_matrix = distance_measure.load_distance_matrix_2d(
            self.sptr_metadata.distances_filename, self.sptr_metadata.region)

        # work on Point at (0, 0)
        distances_0_0_as_region = SpatialRegion(distance_matrix[0].reshape((x_len, y_len)))
        plot_util.plot_discrete_spatial_region(distances_0_0_as_region,
                                               'Distances to point at (0,0)', clusters=False)
        plt.draw()
        distances_0_0_output = 'plots/distances_0_0_{}.eps'.format(self.sptr_metadata.name)
        plt.savefig(distances_0_0_output)
        plt.show()

        # work on Point at center of graph
        center = int(x_len * y_len / 2)
        if x_len % 2 == 0:
            # so far the center variable points to first element of 'center' row, add half row
            center = center + int(y_len / 2)

        distances_center_as_region = SpatialRegion(distance_matrix[center].reshape((x_len, y_len)))
        plot_util.plot_discrete_spatial_region(distances_center_as_region,
                                               'Distances to center point', clusters=False)
        plt.draw()
        distances_center_output = 'plots/distances_center_{}.eps'.format(self.sptr_metadata.name)
        plt.savefig(distances_center_output)
        plt.show()

    def execute_command(self, command):
        self.commands[command]()

    def __load_saved_spt_region(self):
        saved_dataset = np.load(self.sptr_metadata.dataset_filename)
        return SpatioTemporalRegion(saved_dataset)


def do_silhouette_analysis(spt_region, silhouette_metadata):
    '''
    Given a spatio-temporal region, creates silhouette graphs and calculates the silhouette
    average for each provided k, using k-medoids algorithm and provided distance function.
    It requires the distance_measure to have a distance matrix available, so this function will
    compute it if not provided.
    '''

    best_silhouette_avg = -1
    best_k = 0
    best_seed = None
    best_medoids = None
    best_labels = None

    # k-medoids expects a matrix (n_samples x n_features)
    # this converts spatio-temporal region in a list of temporal series
    series_group = spt_region.as_2d

    # we also need the shape for graphs
    (_, x_len, y_len) = spt_region.shape
    shape_2d = (x_len, y_len)

    # need a distance matrix for silhouette analysis
    # this is done once, can be done before-hand
    distance_measure = silhouette_metadata.distance_measure
    if distance_measure.distance_matrix is None:
        distance_measure.compute_distance_matrix(spt_region)

    # iterate seed values
    for random_seed in silhouette_metadata.seed_values:

        logger.info('Using seed: {}'.format(random_seed))

        # iterate k values, this iteration will be repeated for each seed
        # TODO isolate seeds and ks?
        for k in silhouette_metadata.k_values:

            # run k-medoids algorithm
            kmedoids_result = single_kmedoids_for_silhouette(series_group, k, random_seed,
                                                             silhouette_metadata)

            # silhouette is computed and shown here
            silhouette_avg = single_silhouette(kmedoids_result, silhouette_metadata, shape_2d)

            # save best results
            if silhouette_avg > best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_k = k
                best_seed = random_seed
                best_medoids = kmedoids_result.medoids
                best_labels = kmedoids_result.labels

    return best_k, best_seed, best_medoids, best_labels


def single_kmedoids_for_silhouette(series_group, k, random_seed, silhouette_metadata):
    '''
    Runs k-medoids to find the medoids and the clusters.
    The region is already represented as a list of series to save computation time.

    Output is a kmedoids.KmedoidsResult namedtuple
    '''
    kmedoids_metadata = kmedoids_metadata_from_silhouette_metadata(k, random_seed,
                                                                   silhouette_metadata)

    # apply k-medoids on the data
    return kmedoids.run_kmedoids_from_metadata(series_group, kmedoids_metadata)


def single_silhouette(kmedoids_result, silhouette_metadata, shape_2d):
    '''
    Performs silhouette analysis on a spatio temporal region for a given k and seed.
    The region is already represented as a list of series to save computation time.
    The shape of the region is needed for graphs.

    FIXME: graphs are not saved unless they are shown (silhouette_metadata.show_graphs=True)
    '''

    (k, random_seed, labels) = (kmedoids_result.k, kmedoids_result.random_seed,
                                kmedoids_result.labels)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    title = "Silhouette analysis for KMeans clustering on sample data with k={}, seed={}"
    plt.suptitle(title.format(k, random_seed), fontsize=14, fontweight='bold')

    if silhouette_metadata.show_graphs:
        # plot the clustering in 2d
        plot_util.plot_2d_clusters(labels, shape_2d, title='Clustering', subplot=ax1)

    # building the silhouette graph requires all distances
    distance_matrix = silhouette_metadata.distance_measure.distance_matrix
    assert distance_matrix is not None

    # this computes the silhouette average *and* creates the plot
    silhouette_avg = plot_util.plot_clustering_silhouette(distance_matrix, labels, subplot=ax2)

    if silhouette_metadata.show_graphs:
        plt.show()

    save_graphs = silhouette_metadata.save_graphs
    if save_graphs is not None:
        # save the figure for this k
        filename_k = '{}_k{}_seed{}.eps'.format(save_graphs, str(k), str(random_seed))
        fig.savefig(filename_k)

    # the silhouette score
    return silhouette_avg


def kmedoids_metadata_from_silhouette_metadata(k, random_seed, silhouette_metadata):
    '''
    Returns an instance of kmedoids.KmedoidsMetadata given the silhouette metadata and k/seed
    '''

    # if initial_medoids is provided, get the first k medoids
    k_initial_medoids = None
    if silhouette_metadata.initial_medoids is not None:
        k_initial_medoids = silhouette_metadata.initial_medoids[0:k]

    return kmedoids.KmedoidsMetadata(k=k, distance_measure=silhouette_metadata.distance_measure,
                                     initial_medoids=k_initial_medoids,
                                     random_seed=random_seed,
                                     mode=silhouette_metadata.kmedoids_mode,
                                     max_iter=silhouette_metadata.max_iter,
                                     tol=silhouette_metadata.tol,
                                     verbose=silhouette_metadata.verbose)


DEFAULT_SAVE_GRAPHS = 'plots/silhouette'


def silhouette_default_metadata(k_values, seed_values, kmedoids_mode='lite',
                                distance_measure=DistanceByDTW(), initial_medoids=None,
                                max_iter=1000, tol=0.001, verbose=True, show_graphs=True,
                                save_graphs=DEFAULT_SAVE_GRAPHS, plot_name=None):
    '''
    Metadata for Silhouette analysis with default values. Still needs k_values and seed_values.
    Adds plot_name, will define save_graphs if present.
    '''
    if plot_name and save_graphs == DEFAULT_SAVE_GRAPHS:
        # override save_graphs
        save_graphs = 'plots/{}'.format(plot_name)

    return SilhouetteMetadata(k_values=k_values, seed_values=seed_values,
                              kmedoids_mode=kmedoids_mode, distance_measure=distance_measure,
                              initial_medoids=initial_medoids, max_iter=max_iter, tol=tol,
                              verbose=verbose, show_graphs=show_graphs, save_graphs=save_graphs)


if __name__ == '__main__':

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # Run with sp_small
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata

    # use this region as example
    sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)

    # need to specify k/seeds for silhouette analysis
    k_values = [2, 3, 4]
    seed_values = [0, 1]
    silhouette_md = silhouette_default_metadata(k_values=k_values, seed_values=seed_values,
                                                save_graphs='plots/sp_small')

    sp_small_analysis = KmedoidsWithSilhouette(sp_small_md, silhouette_md)
    command_options = sp_small_analysis.commands.keys()

    if len(sys.argv) == 1 or sys.argv[1] not in command_options:

        _, filename = os.path.split(sys.argv[0])
        opts_str = '|'.join(command_options)
        print('Usage: {} [{}]'.format(filename, opts_str))
        sys.exit(1)

    sp_small_analysis.execute_command(sys.argv[1])
