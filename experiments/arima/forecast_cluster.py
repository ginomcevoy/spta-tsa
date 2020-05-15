'''
Execute this program to perform clustering on a spatio-temporal region, then apply ARIMA
forecasting and error evaluation on each cluster.
'''
import argparse

from spta.arima import arima
from spta.distance.dtw import DistanceByDTW
from spta.kmedoids import kmedoids
from spta.region import SpatioTemporalRegion, SpatioTemporalCluster
from spta.util import log as log_util

from experiments.metadata.arima import predefined_arima_experiments
from experiments.metadata.arima_clustering import arima_clustering_experiments
from experiments.metadata.region import predefined_regions


def processRequest():

    # parses the arguments
    desc = 'Perform clustering on a spatial-region, ' \
        'then call arima.evaluate_forecast_errors_arima on each cluster'
    usage = '%(prog)s [-h] <region> <arima_clustering_experiment> [--log=log_level]'
    parser = argparse.ArgumentParser(prog='arima_forecast_cluster', description=desc, usage=usage)

    # for now, need name of region metadata and id of arima_clustering
    region_options = predefined_regions().keys()
    arima_clustering_options = arima_clustering_experiments().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('arima_clustering', help='ID of arima clustering experiment',
                        choices=arima_clustering_options)
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    do_arima_forecast_cluster(args)


def do_arima_forecast_cluster(args):

    # get the region from metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

    # get experiment parameters
    # e.g. 'arima_simple_2_0': ['arima_simple', 'Kmedoids', 2, 0]
    exp_params = arima_clustering_experiments()[args.arima_clustering]
    (arima_experiment_id, clustering_id, distance_id, k, seed) = exp_params
    arima_experiment = predefined_arima_experiments()[arima_experiment_id]

    # for now
    assert clustering_id == 'Kmedoids'
    assert distance_id == 'DistanceByDTW'

    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(spt_region_metadata.distances_filename,
                                         spt_region_metadata.region)

    # build a KmedoidsMetadata object
    kmedoids_metadata = kmedoids.kmedoids_default_metadata(k, distance_measure=distance_dtw,
                                                           random_seed=seed)

    # run k-medoids
    # KmedoidsResult(k, random_seed, medoids, labels, costs, tot_cost, dist_mat)
    kmedoids_result = kmedoids.run_kmedoids_from_metadata(spt_region.as_2d, kmedoids_metadata)
    medoid_indices = kmedoids.get_medoid_indices(kmedoids_result.medoids)

    # build the spatio-temporal clusters
    clusters = []
    for i in range(0, k):
        cluster_i = SpatioTemporalCluster.from_clustering(spt_region, kmedoids_result.labels,
                                                          label=i, centroids=medoid_indices)
        clusters.append(cluster_i)

    # iterate arima analysis, for each analysis work on all clusters
    for arima_params in arima_experiment:

        for cluster_i in clusters:
            centroid_i = cluster_i.centroid
            (centroid, training_region, forecast_region, test_region, arima_region) = \
                arima.evaluate_forecast_errors_arima(cluster_i, arima_params, centroid=centroid_i)


if __name__ == '__main__':
    processRequest()
