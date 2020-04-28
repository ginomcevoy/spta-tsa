import logging
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from spta.region import Region
from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion

from spta.distance.dtw import DistanceByDTW, DistanceBySpatialDTW
from spta.distance.dtw_parallel import DistanceByDTWParallel
from spta.kmedoids import kmedoids, get_medoid_indices
from spta.util import plot as plot_util


# NORDESTE:     20:50, 65:95
# SP_RIO:       40:75, 50:85
# SP:           55:75, 50:70
# BRIAN?:       40:80, 45:85

WHOLE_BRAZIL_REGION = Region(20, 100, 15, 95)
WHOLE_BRAZIL_DATASET = 'raw/whole_brazil1y_1ppd.npy'
WHOLE_BRAZIL_DISTANCES = 'raw/distances_whole_brazil1y_1ppd.npy'

logger = logging.getLogger()


def save_dataset(argv):
    '''
    Saves the dataset into a file
    '''
    dataset_1y_1ppd = SpatioTemporalRegion.load_1year_1ppd_last()
    whole_brazilregion = dataset_1y_1ppd.region_subset(WHOLE_BRAZIL_REGION)
    whole_brazilregion.save(WHOLE_BRAZIL_DATASET)


def calculate_distances(argv):
    num_proc = int(argv[2])

    # load file
    whole_brazildataset = np.load(WHOLE_BRAZIL_DATASET)
    whole_brazilregion = SpatioTemporalRegion(whole_brazildataset)

    logger.info('Calculating distances using DTW...')
    distance_measure = DistanceByDTWParallel(num_proc)
    distance_matrix = distance_measure.compute_distance_matrix(whole_brazilregion)
    np.save(WHOLE_BRAZIL_DISTANCES, distance_matrix)


def silhouette_analysis(argv):

    # Load the dataset
    whole_brazildataset = np.load(WHOLE_BRAZIL_DATASET)
    whole_brazilregion = SpatioTemporalRegion(whole_brazildataset)
    _, x_len, y_len = whole_brazilregion.shape
    print(whole_brazilregion.shape)

    # Run silhouette analysis for range of ks
    ks = range(2, 11)
    seeds = range(0, 4)

    # initial_medoids = [0, 69, 11, 13]
    initial_medoids = None

    # weight = 0.8
    weight = None
    save_graphs = 'plots/whole_brazilweight{}'.format(weight)

    if weight:
        logger.info('Using weighted DTW w={}'.format(weight))
        distance_measure = DistanceBySpatialDTW(weight=weight)
    else:
        logger.info('Using DTW')
        distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    distance_measure.load_distance_matrix_2d(WHOLE_BRAZIL_DISTANCES, WHOLE_BRAZIL_REGION)

    silhouette_result = kmedoids.silhouette_spt(ks, whole_brazilregion, distance_measure,
                                                initial_medoids=initial_medoids, seeds=seeds,
                                                max_iter=1000, tol=0.001, verbose=True,
                                                show_graphs=True, save_graphs=save_graphs)

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
    'save': save_dataset,
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
    logging.getLogger('matplotlib.backends.backend_ps').disabled = True

    # work according to requested command
    FUNC_BY_CMD[sys.argv[1]](sys.argv)
