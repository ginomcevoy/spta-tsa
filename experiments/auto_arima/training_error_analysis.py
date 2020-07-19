'''
Execute this program to perform a suite of k-medoids clustering on a spatio-temporal region.
Then, for each partitioning, apply ARIMA forecasting obtained via auto_arima and do error
analysis on each cluster.
'''
import argparse
import csv
from collections import namedtuple
import os

from spta.arima.forecast import ArimaForecastingAutoArima
from spta.arima import analysis as arima_analysis

from spta.clustering.factory import ClusteringFactory
from spta.distance.dtw import DistanceByDTW
from spta.region.error import error_functions

from spta.util import fs as fs_util
from spta.util import log as log_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.clustering import get_suite as get_clustering_suite
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
    usage = '%(prog)s [-h] <region> <auto_arima_id> [--error error_type] [--parallel n] ' \
        ' [--plots] [--log log_level]'

    parser = argparse.ArgumentParser(prog='arima_forecast_cluster', description=desc, usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # auto_arima_id, see metadata.arima_clustering
    auto_arima_clustering_options = auto_arima_clustering_experiments().keys()
    parser.add_argument('auto_arima_clustering_id', help='ID of auto arima clustering experiment',
                        choices=auto_arima_clustering_options)

    # error type is optional and defaults to sMAPE
    error_options = error_functions().keys()
    error_help_msg = 'error type (default: %(default)s)'
    parser.add_argument('--error', help=error_help_msg, default='sMAPE', choices=error_options)

    # optionally use parallelization
    parser.add_argument('--parallel', help='number of parallel workers')

    # optionally create plots
    parser.add_argument('--plots', help='create relevant plots (distances vs errors)',
                        default=False, action='store_true')

    # logging
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_auto_arima_error_analysis(args, logger)


def do_auto_arima_error_analysis(args, logger):

    # parse to get metadata
    region_metadata, clustering_suite, auto_arima_params, distance = metadata_from_args(args)
    error_type = args.error

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()
    _, x_len, y_len = spt_region.shape

    # assume DTW for now
    assert distance == 'dtw'

    # use parallelization?
    parallel_workers = None
    if args.parallel:
        parallel_workers = int(args.parallel)

    # print plots?
    with_plots = False
    if args.plots:
        with_plots = True

    # use pre-computed distance matrix
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # clustering algorithm to use
        clustering_factory = ClusteringFactory(distance_measure)
        clustering_algorithm = clustering_factory.instance(clustering_metadata)

        do_auto_arima_error_analysis_for_clustering(spt_region, clustering_algorithm,
                                                    auto_arima_params, error_type,
                                                    parallel_workers, with_plots, logger)


def do_auto_arima_error_analysis_for_clustering(spt_region, clustering_algorithm,
                                                auto_arima_params, error_type, parallel_workers,
                                                with_plots, logger, output_prefix='outputs'):

    # use the clustering algorithm to get the partition and medoids
    # also save the partition details (needs region metadata)
    partition, medoid_points = clustering_algorithm.partition(spt_region,
                                                              with_medoids=True,
                                                              save_csv_at=output_prefix)

    clusters = partition.create_all_spt_clusters(spt_region, medoids=medoid_points)

    # create the output dir
    # outputs/<region>/<distance>/<clustering>/arima-<arima_suite_id>
    clustering_output_dir = clustering_algorithm.output_dir(output_prefix,
                                                            spt_region.region_metadata)

    # ensure output dir
    auto_arima_subdir = '{!r}'.format(auto_arima_params)
    output_dir = os.path.join(clustering_output_dir, auto_arima_subdir)
    fs_util.mkdir(output_dir)

    # save results in CSV format: write header now
    # error-analysis__<arima_suite_id>__<clustering>__<error>.csv
    csv_filename = 'error-analysis__{!r}__{!r}__{}.csv'.format(clustering_algorithm,
                                                               auto_arima_params, error_type)
    csv_filepath = os.path.join(output_dir, csv_filename)
    with open(csv_filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(AutoArimaClusterResult._fields)

    # iterate the spatio-temporal clusters
    for i in range(0, clustering_algorithm.k):

        cluster_i = clusters[i]
        analyze_auto_arima_for_cluster(cluster_i, i, clustering_algorithm, auto_arima_params,
                                       error_type, parallel_workers, with_plots, csv_filepath,
                                       output_dir, logger)

    logger.info('CSV output at: {}'.format(csv_filepath))


def analyze_auto_arima_for_cluster(spt_cluster, i, clustering_algorithm, auto_arima_params,
                                   error_type, parallel_workers, with_plots, csv_filepath,
                                   output_dir, logger):

    cluster_medoid = spt_cluster.centroid

    logger.info('*************************************************************')
    logger.info('Analyzing cluster {} with medoid: {}'.format(i, cluster_medoid))
    logger.info('Auto ARIMA: {}'.format(auto_arima_params))
    logger.info('*************************************************************')

    # do the analysis using auto_arima
    forecasting_auto = ArimaForecastingAutoArima(auto_arima_params, parallel_workers)
    analysis_auto = arima_analysis.ArimaErrorAnalysis(forecasting_auto)

    arima_forecasting, overall_errors, forecast_time, compute_time = \
        analysis_auto.evaluate_forecast_errors(spt_cluster, error_type)

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
    with open(csv_filepath, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        logger.info('Writing partial result: {}'.format(experiment_result))
        csv_writer.writerow(experiment_result)

    # plot distances to medoid vs forecast errors using medoid model?
    if with_plots:

        plot_name = plot_name_distances_vs_errors(clustering_algorithm, error_type,
                                                  spt_cluster, p_medoid, d_medoid, q_medoid,
                                                  output_dir)
        plot_desc = '{!r}'.format(clustering_algorithm)
        arima_forecasting.plot_distances_vs_errors(cluster_medoid,
                                                   arima_analysis.FORECAST_LENGTH,
                                                   error_type,
                                                   clustering_algorithm.distance_measure,
                                                   plot_name=plot_name,
                                                   plot_desc=plot_desc)


def metadata_from_args(args):
    '''
    Extract the experiment details from the request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata from auto-ARIMA clustering experiment
    experiment = auto_arima_clustering_experiments()[args.auto_arima_clustering_id]

    clustering_suite = get_clustering_suite(clustering_type=experiment.clustering_type,
                                            suite_id=experiment.clustering_suite_id)

    # get the auto-ARIMA params
    auto_arima_params = predefined_auto_arima()[experiment.auto_arima_id]

    # e.g. DTW
    distance_measure = experiment.distance

    return region_metadata, clustering_suite, auto_arima_params, distance_measure


def plot_name_distances_vs_errors(clustering_algorithm, error_type, cluster, p, d, q,
                                  output_dir):
    '''
    Name and description about distances vs errors plot
    <output_dir>/dist-error__<clustering>__<error>__<cluster>__auto-arima-<arima_params>.pdf
    '''
    plot_name = 'dist-error__{!r}__{}__{}__auto-arima-p{}-d{}-q{}.pdf'.format(clustering_algorithm,
                                                                              error_type,
                                                                              cluster.name,
                                                                              p, d, q)
    plot_filepath = os.path.join(output_dir, plot_name)
    return plot_filepath


if __name__ == '__main__':
    processRequest()
