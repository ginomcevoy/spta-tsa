'''
Execute this program to perform a suite of k-medoids clustering on a spatio-temporal region.
Then, for each partitioning, apply ARIMA forecasting obtained via auto_arima and do error
analysis on each cluster.
'''
import argparse
import csv
from collections import namedtuple

from spta.arima.forecast import ArimaForecastingAutoArima
from spta.arima.analysis import ArimaErrorAnalysis
from spta.distance.dtw import DistanceByDTW
from spta.kmedoids import kmedoids

from spta.region.error import error_functions
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster

from spta.util import fs as fs_util
from spta.util import log as log_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.kmedoids import kmedoids_suites
from experiments.metadata.region import predefined_regions


'''
This tuple will hold analysis results for each cluster.
For each cluster and for each ArimaParams tuple (p, d, q), save the computation time and the
following errors:

    error_each: RMSE of forecast errors using ARIMA models trained in each point

    error_min: RMSE of the forecast errors when using a single ARIMA model to forecast the
        entire region, such that the RMSE is minimized (overall best single ARIMA model)

    error_min_local: RMSE of forecast errors using the ARIMA model that has the
        minimum local error among all points (best local ARIMA model)

    error_medoid: RMSE of forecast errors using the ARIMA model trained at the cluster medoid
        to forecast the entire region (representative ARIMA model)

    error_max: RMSE of the forecast errors when using a single ARIMA model to forecast the
        entire region, such that the RMSE is maximized (overall worst single ARIMA model)
'''
AutoArimaClusterResult = namedtuple('AutoArimaClusterResult',
                                    ('cluster', 'size', 'p_medoid', 'd_medoid', 'q_medoid',
                                     'aic_medoid', 'failed', 't_forecast', 't_elapsed',
                                     'error_each', 'error_min', 'error_min_local',
                                     'error_medoid', 'error_max'))

def processRequest():

    # parses the arguments
    desc = 'Perform various clustering partitions on a spatial-region, ' \
        'then do ARIMA error analysis on each cluster'
    usage = '%(prog)s [-h] <region> <auto_arima_id> <error_type> [--log=log_level]'
    parser = argparse.ArgumentParser(prog='arima_forecast_cluster', description=desc, usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # auto_arima_id, see metadata.arima_clustering
    auto_arima_clustering_options = auto_arima_clustering_experiments().keys()
    parser.add_argument('auto_arima_cluster', help='ID of auto arima clustering experiment',
                        choices=auto_arima_clustering_options)

    # error function
    error_options = error_functions().keys()
    parser.add_argument('error', help='Error type', choices=error_options)

    # optional arguments
    parser.add_argument('--parallel', help='number of parallel workers')
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_auto_arima_forecast_cluster(args, logger)


def do_auto_arima_forecast_cluster(args, logger):

    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the region and transform to list of time series
    spt_region = SpatioTemporalRegion.from_metadata(region_metadata)

    _, _, y_len = spt_region.shape
    series_group = spt_region.as_2d

    # get the auto_arima cluster experiment
    # and from that, the kmedoids suite and auto_arima params
    experiment = auto_arima_clustering_experiments()[args.auto_arima_cluster]
    kmedoids_suite = kmedoids_suites()[experiment.kmedoids_id]

    # Assumption: use DTW
    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(region_metadata.distances_filename,
                                         region_metadata.region)

    # iterate the suite
    for kmedoids_metadata in kmedoids_suite:

        # assumption: overwrite distance_measure...
        # TODO think of a better approach, probably handle this internally to distance_measure
        kmedoids_metadata.distance_measure.distance_matrix = distance_dtw.distance_matrix

        # run K-medoids, this generates a KmedoidsResult namedtuple
        # analyze this particular partition
        kmedoids_result = kmedoids.run_kmedoids_from_metadata(series_group, kmedoids_metadata)
        analyze_kmedoids_partition(spt_region, kmedoids_result, experiment, args, logger)


def analyze_kmedoids_partition(spt_region, kmedoids_result, experiment, args, logger):

    k, random_seed = kmedoids_result.k, kmedoids_result.random_seed
    auto_arima_params = predefined_auto_arima()[experiment.auto_arima_id]

    # build the spatio-temporal clusters
    clusters = []
    for i in range(0, k):
        cluster_i = SpatioTemporalCluster.from_crisp_clustering(spt_region, kmedoids_result.labels,
                                                                cluster_index=i,
                                                                centroids=kmedoids_result.medoids)
        clusters.append(cluster_i)

    # prepare the CSV output
    # csv/<k>/auto_arima_<auto_arima_id>_<region>_kmedoids_<k>_<seed>.csv
    output_dir = 'csv/{}'.format(k)
    fs_util.mkdir(output_dir)
    csv_file_str = '{}/auto_arima_{}_{}_kmedoids_k{}_seed{}_{}.csv'
    csv_filename = csv_file_str.format(output_dir, experiment.auto_arima_id, args.region, k,
                                       random_seed, args.error)

    # write header now
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(AutoArimaClusterResult._fields)

    # iterate the spatio-temporal clusters
    for i in range(0, k):

        cluster_i = clusters[i]
        analyze_auto_arima_for_cluster(cluster_i, i, auto_arima_params, csv_filename, args, logger)

    logger.info('CSV output of k={}, seed={} at: {}'.format(k, random_seed, csv_filename))

def analyze_auto_arima_for_cluster(spt_cluster, i, auto_arima_params, csv_filename, args, logger):

    cluster_medoid = spt_cluster.centroid

    logger.info('*************************************************************')
    logger.info('Analyzing cluster {} with medoid: {}'.format(i, cluster_medoid))
    logger.info('Auto ARIMA: {}'.format(auto_arima_params))
    logger.info('*************************************************************')

    # use parallelization?
    parallel_workers = None
    if args.parallel:
        parallel_workers = int(args.parallel)

    # do the analysis using auto_arima
    forecasting_auto = ArimaForecastingAutoArima(auto_arima_params, parallel_workers)
    analysis_auto = ArimaErrorAnalysis(forecasting_auto)

    arima_forecasting, overall_errors, forecast_time, compute_time = \
        analysis_auto.evaluate_forecast_errors(spt_cluster, args.error)

    # access the generated ARIMA models
    arima_models = arima_forecasting.arima_models

    # (p_medoid, d_medoid, q_medoid): the (p, d, q) hyper-parameters found by auto_arima
    # at the medoid of this cluster
    # aic_medoid: the aic value obtained by fitting the model at the medoid of this cluster
    (p_medoid, d_medoid, q_medoid) = arima_models.pdq_region.series_at(cluster_medoid)
    aic_medoid = arima_models.aic_region.value_at(cluster_medoid)
    aic_medoid = '{:.3f}'.format(aic_medoid)

    # failed: number of failed models in the cluster
    failed = arima_models.missing_count

    # format timings as nice strings
    t_forecast = '{:.3f}'.format(forecast_time)
    t_elapsed = '{:.3f}'.format(compute_time)

    # format all errors as nice strings
    error_each = '{:.3f}'.format(overall_errors.each)
    error_min = '{:.3f}'.format(overall_errors.minimum)
    error_min_local = '{:.3f}'.format(overall_errors.min_local)
    error_medoid = '{:.3f}'.format(overall_errors.centroid)
    error_max = '{:.3f}'.format(overall_errors.maximum)

    # save error and performance data for this experiment
    experiment_result = AutoArimaClusterResult(cluster=i,
                                               size=spt_cluster.cluster_len,
                                               p_medoid=p_medoid,
                                               d_medoid=d_medoid,
                                               q_medoid=q_medoid,
                                               aic_medoid=aic_medoid,
                                               failed=failed,
                                               t_forecast=t_forecast,
                                               t_elapsed=t_elapsed,
                                               error_each=error_each,
                                               error_min=error_min,
                                               error_min_local=error_min_local,
                                               error_medoid=error_medoid,
                                               error_max=error_max)

    # partial results in CSV format
    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        logger.info('Writing partial result: {}'.format(experiment_result))
        csv_writer.writerow(experiment_result)


if __name__ == '__main__':
    processRequest()
