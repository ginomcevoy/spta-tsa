'''
Execute this program to perform clustering on a spatio-temporal region, then apply ARIMA
forecasting and error evaluation on each cluster.
'''
import argparse
from collections import namedtuple
import csv
import os

from spta.arima.forecast import ArimaForecastingPDQ
from spta.arima import analysis as arima_analysis

from spta.clustering.factory import ClusteringFactory
from spta.distance.dtw import DistanceByDTW
from spta.model.error import error_functions

from spta.util import fs as fs_util
from spta.util import log as log_util

from experiments.metadata.arima import arima_suite_by_name
from experiments.metadata.arima_clustering import experiments_for_arima_with_clusters
from experiments.metadata.clustering import get_suite as get_clustering_suite
from experiments.metadata.region import predefined_regions


'''
This tuple will hold analysis results for each cluster.
For each cluster and for each ArimaParams tuple (p, d, q), save the computation time and the
following errors:

    error_each: RMSE of forecast MASE errors using ARIMA models trained in each point

    error_min: RMSE of the forecast MASE errors when using a single ARIMA model to forecast the
        entire region, such that the RMSE is minimized (overall best single ARIMA model)

    error_min_local: RMSE of forecast MASE errors using the ARIMA model that has the
        minimum local error among all points (best local ARIMA model)

    error_medoid: RMSE of forecast MASE errors using the ARIMA model trained at the cluster medoid
        to forecast the entire region (representative ARIMA model)

    error_max: RMSE of the forecast MASE errors when using a single ARIMA model to forecast the
        entire region, such that the RMSE is maximized (overall worst single ARIMA model)
'''
ArimaClusterResult = namedtuple('ArimaClusterResult',
                                ('cluster', 'size', 'p', 'd', 'q', 'failed', 't_forecast',
                                 't_elapsed', 'error_each', 'error_min', 'error_min_local',
                                 'error_medoid', 'error_max'))

def processRequest():

    # parses the arguments
    desc = 'Perform clustering on a spatial-region, ' \
        'then call arima.evaluate_forecast_errors_arima on each cluster'
    usage = '%(prog)s [-h] <region> <arima_clustering_experiment> [--error <error_type>] ' \
        '[--log=log_level]'
    parser = argparse.ArgumentParser(prog='arima_forecast_cluster', description=desc, usage=usage)

    # for now, need name of region metadata and id of arima_clustering
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    arima_clustering_options = experiments_for_arima_with_clusters().keys()
    parser.add_argument('arima_clustering_id', help='ID of arima clustering experiment',
                        choices=arima_clustering_options)

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
    do_arima_error_analysis(args, logger)


def do_arima_error_analysis(args, logger):

    # parse to get metadata
    region_metadata, clustering_suite, arima_suite, distance = metadata_from_args(args)
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

        do_arima_error_analysis_for_clustering(spt_region, clustering_algorithm, arima_suite,
                                               error_type, parallel_workers, with_plots, logger)


def do_arima_error_analysis_for_clustering(spt_region, clustering_algorithm, arima_suite,
                                           error_type, parallel_workers, with_plots, logger,
                                           output_home='outputs'):

    # use the clustering algorithm to get the partition and medoids
    # also save the partition details (needs region metadata)
    # will try to leverage pickle and load previous attempts, otherwise calculate and save
    partition = clustering_algorithm.partition(spt_region,
                                               with_medoids=True,
                                               save_csv_at=output_home,
                                               pickle_home='pickle')

    clusters = partition.create_all_spt_clusters(spt_region, medoids=partition.medoids)

    # create the output dir
    # outputs/<region>/<distance>/<clustering>/arima-<arima_suite_id>
    clustering_output_dir = clustering_algorithm.output_dir(output_home,
                                                            spt_region.region_metadata)

    # ensure output dir
    arima_subdir = 'arima-{}'.format(arima_suite.name)
    output_dir = os.path.join(clustering_output_dir, arima_subdir)
    fs_util.mkdir(output_dir)

    # save results in CSV format: write header now
    # error-analysis__<clustering>__<arima_suite_id>__<error>.csv
    csv_filename = 'error-analysis__{!r}__{}__{}.csv'.format(clustering_algorithm,
                                                             arima_suite.name, error_type)
    csv_filepath = os.path.join(output_dir, csv_filename)
    with open(csv_filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(ArimaClusterResult._fields)

    # iterate the spatio-temporal clusters
    for i in range(0, clustering_algorithm.k):

        # the medoid will be used as centroid for the ARIMA analysis
        cluster_i = clusters[i]
        centroid_i = cluster_i.centroid
        size_i = cluster_i.cluster_len

        logger.info('************************************')
        logger.info('Analyzing cluster {} with medoid: {}'.format(i, centroid_i))
        logger.info('************************************')

        # iterate the ARIMA suite
        for arima_params in arima_suite.arima_params_gen():

            # do the analysis with current ARIMA hyper-parameters
            forecasting_pdq = ArimaForecastingPDQ(arima_params, parallel_workers=parallel_workers)
            analysis_pdq = arima_analysis.ArimaErrorAnalysis(forecasting_pdq)

            arima_forecasting, overall_errors, forecast_time, compute_time = \
                analysis_pdq.evaluate_forecast_errors(cluster_i, error_type)

            # prepare to save experiment result
            t_forecast = '{:.3f}'.format(forecast_time)
            t_elapsed = '{:.3f}'.format(compute_time)
            (p, d, q) = arima_params
            failed = arima_forecasting.arima_models.missing_count

            # format all errors as nice strings
            error_each = '{:.3f}'.format(overall_errors.each)
            error_min = '{:.3f}'.format(overall_errors.minimum)
            error_min_local = '{:.3f}'.format(overall_errors.min_local)
            error_medoid = '{:.3f}'.format(overall_errors.centroid)
            error_max = '{:.3f}'.format(overall_errors.maximum)

            # save error and performance data for this experiment
            arima_experiment = ArimaClusterResult(i, size_i, p, d, q, failed, t_forecast,
                                                  t_elapsed, error_each, error_min,
                                                  error_min_local, error_medoid, error_max)

            # partial results in CSV format
            with open(csv_filepath, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                        quoting=csv.QUOTE_MINIMAL)
                logger.info('Writing partial result: {}'.format(arima_experiment))
                csv_writer.writerow(arima_experiment)

            # plot distances to medoid vs forecast errors using medoid model?
            if with_plots:

                plot_name = plot_name_distances_vs_errors(clustering_algorithm, error_type,
                                                          cluster_i, arima_params, output_dir)
                plot_desc = '{!r}'.format(clustering_algorithm)
                arima_forecasting.plot_distances_vs_errors(centroid_i,
                                                           arima_analysis.FORECAST_LENGTH,
                                                           error_type,
                                                           clustering_algorithm.distance_measure,
                                                           plot_name=plot_name,
                                                           plot_desc=plot_desc)

    logger.info('CSV output at: {}'.format(csv_filepath))


def metadata_from_args(args):
    '''
    Extract the experiment details from the request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata from ARIMA clustering experiment
    arima_clustering_id = experiments_for_arima_with_clusters()[args.arima_clustering_id]
    clustering_suite = get_clustering_suite(arima_clustering_id.clustering_type,
                                            arima_clustering_id.clustering_suite_id)

    # get the arima suite, patch in the name of the suite
    arima_suite = arima_suite_by_name(arima_clustering_id.arima_suite_id)
    arima_suite.name = arima_clustering_id.arima_suite_id

    # e.g. DTW
    distance_measure = arima_clustering_id.distance

    return region_metadata, clustering_suite, arima_suite, distance_measure


def plot_name_distances_vs_errors(clustering_algorithm, error_type, cluster, arima_params,
                                  output_dir):
    '''
    Name and description about distances vs errors plot
    <output_dir>/dist-error__<clustering>__<error>__<cluster>__arima-<arima_params>.pdf
    '''
    p, d, q = arima_params.p, arima_params.d, arima_params.q
    plot_name = 'dist-error__{!r}__{}__{}__arima-p{}-d{}-q{}.pdf'.format(clustering_algorithm,
                                                                         error_type, cluster.name,
                                                                         p, d, q)
    plot_filepath = os.path.join(output_dir, plot_name)
    return plot_filepath


if __name__ == '__main__':
    processRequest()
