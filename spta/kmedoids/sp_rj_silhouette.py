'''
Silhouette analysis for SP_RJ region
'''
import logging
import numpy as np
import matplotlib.pyplot as plt

from spta.dataset import sp_rj
from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.region.distance import DistanceByDTW
from spta.util import plot as plot_util

from . import kmedoids, get_medoid_indices


def main():
    logger = logging.getLogger()

    # Load the dataset
    sp_rj_dataset = np.load(sp_rj.SP_RJ_DATASET)
    sp_rj_region = SpatioTemporalRegion(sp_rj_dataset)
    x_len, y_len, _ = sp_rj_region.shape

    # Use DTW, use pre-computed distance matrix
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_2d(sp_rj.SP_RJ_DISTANCES, sp_rj.SP_RJ_REGION)

    # Run silhouette analysis for range of ks
    ks = range(2, 15)
    best_k, best_medoids, best_labels = kmedoids.silhouette_spt(ks, sp_rj_region, distance_measure,
                                                                seed=3, max_iter=1000, tol=0.001,
                                                                verbose=False, with_graphs=True)

    # Show best results
    logger.info('Best k: {}'.format(best_k))
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


if __name__ == '__main__':
    log_level = logging.INFO
    # log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    main()
