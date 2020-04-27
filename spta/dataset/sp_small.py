import logging
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from spta.region import Region
from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.region.distance import DistanceByDTW
from spta.kmedoids import kmedoids, get_medoid_indices
from spta.util import plot as plot_util


# NORDESTE:     20:50, 65:95
# SP_RIO:       40:75, 50:85
# SP:           55:75, 50:70
# BRIAN?:       40:80, 45:85

SP_SMALL_REGION = Region(40, 50, 50, 60)
SP_SMALL_DATASET = 'raw/sp_small_1y_4ppd.npy'
SP_SMALL_DISTANCES = 'raw/distances_sp_small_1y_4ppd.npy'

logger = logging.getLogger()


def save_sp_small_dataset():
    '''
    Saves the SP/RJ dataset into a file
    '''
    dataset_1y_1ppd = SpatioTemporalRegion.load_1year_last()
    sp_small_region = dataset_1y_1ppd.region_subset(SP_SMALL_REGION)
    sp_small_region.save(SP_SMALL_DATASET)


def calculate_distances():
    # load file
    sp_small_dataset = np.load(SP_SMALL_DATASET)
    sp_small_region = SpatioTemporalRegion(sp_small_dataset)

    logger.info('Calculating distances using DTW...')
    distance_measure = DistanceByDTW()
    distance_matrix = distance_measure.compute_distance_matrix(sp_small_region)
    np.save(SP_SMALL_DISTANCES, distance_matrix)


def silhouette_analysis():

    # Load the dataset
    sp_small_dataset = np.load(SP_SMALL_DATASET)
    sp_small_region = SpatioTemporalRegion(sp_small_dataset)
    _, x_len, y_len = sp_small_region.shape
    print(sp_small_region.shape)

    # Use DTW, use pre-computed distance matrix
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_2d(SP_SMALL_DISTANCES, SP_SMALL_REGION)

    # Run silhouette analysis for range of ks
    ks = range(2, 8)
    seeds = range(0, 8)

    # initial_medoids = [0, 69, 11, 13]
    initial_medoids = None

    silhouette_result = kmedoids.silhouette_spt(ks, sp_small_region, distance_measure,
                                                initial_medoids=initial_medoids, seeds=seeds,
                                                max_iter=1000, tol=0.001, verbose=True,
                                                show_graphs=True,
                                                save_graphs='plots/sp_small')

    best_k, best_seed, best_medoids, best_labels = silhouette_result

    # Show best results
    logger.info('Best k: {}'.format(best_k))
    logger.info('Best seed: {}'.format(best_seed))
    logger.info('Best medoids: {}'.format(str(get_medoid_indices(best_medoids))))
    logger.info('Best labels: {}'.format(str(best_labels)))

    # print results for best k
    # Create a subplot with 1 row and 2 columns
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % best_k), fontsize=14, fontweight='bold')

    label_region = SpatialRegion.create_from_1d(best_labels, x_len, y_len)
    plot_util.plot_discrete_spatial_region(label_region, 'Best output mask', subplot=ax1)

    # build the silhouette graph, requires all distances
    plot_util.plot_clustering_silhouette(distance_measure.distance_matrix,
                                         best_labels, subplot=ax2)


FUNC_BY_CMD = {
    'save': save_sp_small_dataset,
    'distances': calculate_distances,
    'kmedoids': silhouette_analysis
}

CMD_OPTIONS = FUNC_BY_CMD.keys()


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in CMD_OPTIONS:

        _, filename = os.path.split(sys.argv[0])
        opts_str = '|'.join(CMD_OPTIONS)
        print('Usage: {} [{}]'.format(filename, opts_str))
        sys.exit(1)

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger('matplotlib.font_manager').disabled = True

    # work according to requested command
    FUNC_BY_CMD[sys.argv[1]]()
