'''
Execute this program to perform clustering on a spatio-temporal region, then apply ARIMA
forecasting and error evaluation on each cluster.
'''
import argparse
import csv
from collections import namedtuple
import time

from spta.arima import arima
from spta.distance.dtw import DistanceByDTW
from spta.kmedoids import kmedoids
from spta.region import SpatioTemporalRegion, SpatioTemporalCluster
from spta.util import log as log_util

from experiments.metadata.arima import arima_suite_by_name
from experiments.metadata.arima_clustering import arima_clustering_experiments
from experiments.metadata.region import predefined_regions


'''
This tuple will hold analysis results for each cluster.
For each cluster and for each ArimaParams tuple (p, d, q), save the computation time and the
following errors:

    error_each: RMSE of forecast MASE errors using ARIMA models trained in each point

    error_min_local: RMSE of forecast MASE errors using the ARIMA model that has the
        minimum local error among all points

    error_medoid: RMSE of forecast MASE errors using the ARIMA model trained at the cluster medoid
'''
ArimaClusterResult = namedtuple('ArimaClusterResult',
                                ('cluster', 'size', 'p', 'd', 'q', 'failed', 't_forecast',
                                 't_elapsed', 'error_each', 'error_min', 'error_min_local',
                                 'error_medoid', 'error_max'))

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
    parser.add_argument('--parallel', help='number of parallel workers')
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')
    parser.add_argument('--plot', help='add the plot of the ARIMA model result at Point(0, 0)?',
                        default=False, action='store_true')

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    do_arima_forecast_cluster(args)


def do_arima_forecast_cluster(args):

    logger = log_util.logger_for_me(do_arima_forecast_cluster)

    # get the region from metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

    # get experiment parameters including ARIMA suite
    exp_params = arima_clustering_experiments()[args.arima_clustering]
    (arima_experiment_id, clustering_id, distance_id, k, seed) = exp_params
    arima_suite = arima_suite_by_name(arima_experiment_id)

    # use parallelization?
    parallel_workers = None
    if args.parallel:
        parallel_workers = int(args.parallel)

    # for now
    assert clustering_id == 'Kmedoids'
    assert distance_id == 'DistanceByDTW'

    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(spt_region_metadata.distances_filename,
                                         spt_region_metadata.region)

    # faster... (whole_brazil_1y_1ppd, k=8, seed=0)
    #initial_medoids = [5816, 1163, 4295, 4905, 3156, 2648, 172, 3764]
    # initial_medoids = [3764, 2648, 5816, 4905, 4295, 172, 1163, 3156]
    initial_medoids = None

    # build a KmedoidsMetadata object
    kmedoids_metadata = kmedoids.kmedoids_default_metadata(k, distance_measure=distance_dtw,
                                                           random_seed=seed,
                                                           initial_medoids=initial_medoids)

    # run k-medoids
    # KmedoidsResult(k, random_seed, medoids, labels, costs, tot_cost, dist_mat)
    kmedoids_result = kmedoids.run_kmedoids_from_metadata(spt_region.as_2d, kmedoids_metadata)
    medoid_indices = kmedoids.get_medoid_indices(kmedoids_result.medoids)

    # build the spatio-temporal clusters
    clusters = []
    for i in range(0, k):
        cluster_i = SpatioTemporalCluster.from_crisp_clustering(spt_region, kmedoids_result.labels,
                                                                cluster_index=i,
                                                                centroids=medoid_indices)
        clusters.append(cluster_i)

    # save results in CSV format: write header now
    csv_filename = '{}.csv'.format(args.arima_clustering)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(ArimaClusterResult._fields)

    # iterate the spatio-temporal clusters
    for i in range(0, k):
    # for i in range(2, 3):

        # the medoid will be used as centroid for the ARIMA analysis
        cluster_i = clusters[i]
        centroid_i = cluster_i.centroid
        size_i = cluster_i.cluster_len

        logger.info('************************************')
        logger.info('Analyzing cluster {} with medoid: {}'.format(i, centroid_i))
        logger.info('************************************')

        # iterate the ARIMA suite
        for arima_params in arima_suite.arima_params_gen():

            # evaluate this ARIMA model
            # this can take some time, register the time as a performance metric
            t_start = time.time()
            arima_result = \
                arima.evaluate_forecast_errors_arima(cluster_i, arima_params,
                                                     centroid=centroid_i,
                                                     parallel_workers=parallel_workers)
            (centroid, training_region, forecast_region_each, test_region, arima_models_each,
                overall_errors, time_forecast) = arima_result

            t_stop = time.time()

            # prepare to save experiment result
            t_forecast = '{:.3f}'.format(time_forecast)
            t_elapsed = '{:.3f}'.format(t_stop - t_start)
            (p, d, q) = arima_params
            failed = arima_models_each.missing_count

            # format all errors as nice strings
            # TODO do a loop?
            (overall_error_each, overall_error_min, overall_error_min_local,
                overall_error_centroid, overall_error_max) = overall_errors

            error_each = '{:.3f}'.format(overall_error_each)
            error_min = '{:.3f}'.format(overall_error_min)
            error_min_local = '{:.3f}'.format(overall_error_min_local)
            error_medoid = '{:.3f}'.format(overall_error_centroid)
            error_max = '{:.3f}'.format(overall_error_max)

            # save error and performance data for this experiment
            arima_experiment = ArimaClusterResult(i, size_i, p, d, q, failed, t_forecast,
                                                  t_elapsed, error_each, error_min,
                                                  error_min_local, error_medoid, error_max)

            # partial results in CSV format
            with open(csv_filename, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                        quoting=csv.QUOTE_MINIMAL)
                logger.info('Writing partial result: {}'.format(arima_experiment))
                csv_writer.writerow(arima_experiment)

            if args.plot:
                # plot forecast at Point(0, 0)
                arima.plot_one_arima(training_region, forecast_region_each, test_region,
                                     arima_models_each)


if __name__ == '__main__':
    processRequest()
