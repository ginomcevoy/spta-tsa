'''
Execute this program to:

1. Partition a spatio-temporal region using a clustering algorithm (e.g. k-medoids)
2. Train auto ARIMA models at the medoids of the resulting clusters.
3. For each cluster, calculate the distance (e.g. DTW) between each point and the corresponding
   medoid, and the generalization error (forecast error using training data) when using the auto
   ARIMA model at the medoid to forecast the series of each point.
4. Plot the distances vs the forecast errors, and save the values as CSV.
'''

import argparse
import csv
import numpy as np
import os

from spta.arima.train import TrainerAutoArima, extract_pdq
from spta.distance.dtw import DistanceByDTW

from spta.model.error import ErrorAnalysis, error_functions
from spta.model.train import SplitTrainingAndTestLast
from spta.solver.train import SolverTrainer

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.clustering import get_suite
from experiments.metadata.region import predefined_regions


def processRequest():

    # parses the arguments
    desc = '''Given a partitioned spatio-temporal region and a clustering algorithm, train auto
ARIMA models at the medoids of the resulting clusters. Then analyze distances vs. forecast errors
for each cluster.'''

    usage = '%(prog)s [-h] <region> <auto_arima_cluster_id> [--flen <forecast_length>] ' \
        '[--error <error_type>] [--plots] [--log <log_level>]'
    parser = argparse.ArgumentParser(prog='auto-arima-distance-errors', description=desc,
                                     usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # auto_arima_cluster_id, see metadata.arima_clustering
    auto_arima_clustering_options = auto_arima_clustering_experiments().keys()
    parser.add_argument('auto_arima_cluster', help='ID of auto arima clustering experiment',
                        choices=auto_arima_clustering_options)

    # flen (forecast length) is optional and defaults to 8
    forecast_help_msg = 'number of samples for forecast/testing (default: %(default)s)'
    parser.add_argument('--flen', help=forecast_help_msg, default=8, type=int)

    # error type is optional and defaults to sMAPE
    error_options = error_functions().keys()
    error_help_msg = 'error type (default: %(default)s)'
    parser.add_argument('--error', help=error_help_msg, default='sMAPE', choices=error_options)

    # plots are optional (CSV output is always generated)
    parser.add_argument('--plots', help='Create distance vs errors plots',
                        default=False, action='store_true')

    # other optional arguments
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_auto_arima_distance_errors(args, logger)


def do_auto_arima_distance_errors(args, logger):

    logger.debug(args)

    # parse to get metadata
    region_metadata, clustering_suite, auto_arima_params = metadata_from_args(args)
    forecast_len = args.flen
    error_type = args.error

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()

    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # prepare to train a solver based on auto ARIMA
        # the training requires a number of samples used as test data (data retained from the model
        # in order to calculate forecast error), this is test_len (--tp)
        model_trainer = TrainerAutoArima(auto_arima_params, region_metadata.x_len, region_metadata.y_len)
        solver_trainer = SolverTrainer(region_metadata=region_metadata,
                                       clustering_metadata=clustering_metadata,
                                       distance_measure=DistanceByDTW(),
                                       model_trainer=model_trainer,
                                       model_params=auto_arima_params,
                                       test_len=forecast_len,
                                       error_type=error_type)

        # use the trainer to get the cluster partition and corresponding medoids
        # will try to leverage pickle and load previous attempts, otherwise calculate and save
        solver_trainer.prepare_for_training()
        partition = solver_trainer.clustering_algorithm.partition(spt_region,
                                                                  with_medoids=True,
                                                                  save_csv_at='outputs',
                                                                  pickle_home='pickle')

        # use the partition to get k clusters
        clusters = partition.create_all_spt_clusters(spt_region, medoids=partition.medoids)

        for cluster in clusters:

            # will do distance vs errors and errors between medoids
            distance_vs_errors_for_cluster(cluster, solver_trainer, partition, forecast_len, error_type,
                                           args.plots, logger)


def distance_vs_errors_for_cluster(cluster, solver_trainer, partition, forecast_len, error_type,
                                   with_plots, logger, output_home='outputs'):

    _, x_len, y_len = cluster.shape

    logger.debug('Cluster has scaling: {}'.format(cluster.has_scaling()))

    # split cluster in training/test cluster regions
    # assuming forecast length = test length
    splitter = SplitTrainingAndTestLast(forecast_len)
    (training_cluster, observation_cluster) = splitter.split(cluster)
    training_cluster.centroid = cluster.centroid

    # get the distances of each point in the cluster to its medoid
    distance_measure = solver_trainer.distance_measure
    distances_to_medoid = distance_measure.distances_to_point(cluster, cluster.centroid,
                                                              cluster.all_point_indices)

    # compute the forecast error (generalization error) when using the model at the medoid
    # the trainer almost has the appropriate implementation, just need to pass a list of medoids
    # instead of just one.
    arima_model_region = solver_trainer.train_models_at_medoids(training_region=training_cluster,
                                                                medoids=(cluster.centroid,))

    # this will evaluate the forecast errors
    error_analysis = ErrorAnalysis(observation_cluster, training_region=training_cluster,
                                   parallel_workers=None)

    # use the model at the medoid to create a forecast
    # the model does not require any other parameters, but needs forecast_len set
    arima_model_region.forecast_len = forecast_len
    arima_at_medoid = arima_model_region.function_at(cluster.centroid)
    forecast_series = arima_at_medoid(None)

    # debugging only
    np.set_printoptions(precision=3)
    obs_debug_msg = 'Observation series at medoid: {}'
    fct_debug_msg = 'Forecast series from medoid: {}'
    logger.debug(obs_debug_msg.format(observation_cluster.series_at(cluster.centroid)))
    logger.debug(fct_debug_msg.format(forecast_series))

    # use extract_pdq to get the pdq values calculated by autoARIMA
    # need these for the CSV filename
    arima_results_at_medoid = arima_model_region.value_at(cluster.centroid)
    (p_medoid, d_medoid, q_medoid) = extract_pdq(arima_results_at_medoid)

    # calculate the forecast error in the entire region, then for each medoid
    # TODO handle normalization somewhere!
    # ideally inside ErrorAnalysis
    error_region = error_analysis.with_repeated_forecast(forecast_series, error_type)

    # get the error values for the region, this also works on clusters
    # useful if plotting so always do it
    forecast_errors = [
        forecast_error
        for __, forecast_error
        in error_region
    ]

    # prepare the CSV output at:
    # outputs/<region>/<distance>/<clustering>/<auto_arima_params>
    # dist-error__<clustering>__<error>__<cluster>__auto-arima-<arima_params>.csv
    csv_dir = solver_trainer.metadata.output_dir(output_home)
    fs_util.mkdir(csv_dir)

    # csv_filename = '{!r}_distance_errors_cluster_{}.csv'.format(trainer.metadata.model_params,
    #                                                             cluster.name)
    output_template = 'dist-error__{!r}__{}__{}__auto-arima-p{}-d{}-q{}.{}'
    csv_filename = output_template.format(solver_trainer.clustering_metadata, error_type,
                                          cluster.name, p_medoid, d_medoid, q_medoid, 'csv')
    csv_full_path = os.path.join(csv_dir, csv_filename)

    # the CSV has this format:
    # <point> <distance(point, medoid)> <forecast error of ARIMA at medoid with point test data>
    with open(csv_full_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        # write header
        distance_str = 'distance_{}'.format(solver_trainer.distance_measure)
        csv_writer.writerow(['position', distance_str, 'forecast_error'])

        # each point in the cluster is a tuple in the CSV
        for i, (point_in_cluster, value) in enumerate(cluster):

            # transform the point to absolute coordinates (from original dataset)
            coords = solver_trainer.region_metadata.absolute_position_of_point(point_in_cluster)

            # to get the distance, use the point ordering
            # this was the same ordering used for calculating the distances, so it's OK
            distance_to_medoid = distances_to_medoid[i]
            distance_to_medoid_str = '{:.3f}'.format(distance_to_medoid)

            # recover the forecast error using same point ordering
            forecast_error = forecast_errors[i]
            forecast_error_str = '{:.3f}'.format(forecast_error)

            csv_writer.writerow([coords, distance_to_medoid_str, forecast_error_str])

    logger.info('Wrote CSV for {} at: {}'.format(cluster, csv_full_path))

    # plot distances vs forecast errors?
    if with_plots:

        # get ARIMA order at point of interest, need the value to get the actual ARIMA object
        fitted_arima_at_medoid = arima_model_region.value_at(cluster.centroid)
        arima_order = fitted_arima_at_medoid.model.order

        # add some info about plot
        info_text = '\n'.join((
            '{}'.format(cluster),
            'ARIMA: {}'.format(arima_order),
            '{}'.format(solver_trainer.clustering_metadata)))

        # save the plot
        # outputs/<region>/<distance>/<clustering>/<auto_arima_params>
        # dist-error__<clustering>__<error>__<cluster>__auto-arima-<arima_params>.pdf
        plot_dir = solver_trainer.metadata.output_dir(output_home)
        # plot_name = '{!r}_distance_errors_cluster_{}.pdf'.format(trainer.metadata.model_params,
        #                                                          cluster.name)
        output_template = 'dist-error__{!r}__{}__{}__auto-arima-p{}-d{}-q{}.{}'
        plot_name = output_template.format(solver_trainer.clustering_metadata, error_type,
                                           cluster.name, p_medoid, d_medoid, q_medoid, 'pdf')
        plot_full_path = os.path.join(plot_dir, plot_name)

        plot_util.plot_distances_vs_forecast_errors(distances_to_point=distances_to_medoid,
                                                    forecast_errors=forecast_errors,
                                                    distance_measure=distance_measure,
                                                    error_type=error_type,
                                                    info_text=info_text,
                                                    plot_filename=plot_full_path)


def metadata_from_args(args):
    '''
    Metadata common to both train and predict.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata from the auto arima clustering id
    auto_arima_cluster = auto_arima_clustering_experiments()[args.auto_arima_cluster]
    clustering_type = auto_arima_cluster.clustering_type
    clustering_suite = get_suite(clustering_type, auto_arima_cluster.clustering_suite_id)

    # get the auto arima params
    auto_arima_params = predefined_auto_arima()[auto_arima_cluster.auto_arima_id]

    return region_metadata, clustering_suite, auto_arima_params


if __name__ == '__main__':
    processRequest()
