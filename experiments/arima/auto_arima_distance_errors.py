'''
Execute this program to perform ARIMA forecasting and error evaluation on an entire region.
'''
import argparse
import csv
import numpy as np
import os

from spta.distance.dtw import DistanceByDTW

from spta.region.error import ErrorAnalysis, error_functions
from spta.region.train import SplitTrainingAndTestLast

from spta.solver.auto_arima import AutoARIMATrainer

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.clustering import get_suite
from experiments.metadata.region import predefined_regions


def processRequest():

    # parses the arguments
    desc = 'Perform various clustering partitions on a spatial-region, and analyze distances ' \
        'vs. forecast errors for each cluster.'
    usage = '%(prog)s [-h] <region> <auto_arima_cluster_id> [--forecast_len <int>] ' \
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

    # forecast_len is optional and defaults to 8
    forecast_help_msg = 'number of samples for forecast/testing (default: %(default)s)'
    parser.add_argument('--forecast_len', help=forecast_help_msg, default=8, type=int)

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
    forecast_len = args.forecast_len
    error_type = args.error

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()

    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # prepare to train a solver based on auto ARIMA
        trainer = AutoARIMATrainer(region_metadata=region_metadata,
                                   clustering_metadata=clustering_metadata,
                                   distance_measure=DistanceByDTW(),
                                   auto_arima_params=auto_arima_params)

        # use the trainer to get the partition (PartitionRegion) via the cluster algorithm
        trainer.prepare_for_training()
        partition, medoids = trainer.clustering_algorithm.partition(spt_region, with_medoids=True)

        # use the partition to get k clusters
        clusters = partition.create_all_spt_clusters(spt_region, medoids=medoids)

        for cluster in clusters:
            do_auto_arima_distance_errors_for_cluster(cluster, trainer, partition, forecast_len,
                                                      error_type, args.plots, logger)


def do_auto_arima_distance_errors_for_cluster(cluster, trainer, partition, forecast_len,
                                              error_type, with_plots, logger):

    _, x_len, y_len = cluster.shape

    # split cluster in training/test cluster regions
    # assuming forecast length = test length
    splitter = SplitTrainingAndTestLast(forecast_len)
    (training_cluster, observation_cluster) = splitter.split(cluster)
    training_cluster.centroid = cluster.centroid

    # get the distances of each point in the cluster to its medoid
    distance_measure = trainer.distance_measure
    distances_to_medoid = distance_measure.distances_to_point(cluster, cluster.centroid,
                                                              cluster.all_point_indices)

    # compute the forecast error (generalization error) when using the model at the medoid
    # the trainer has the appropriate implementation, but needs a numpy to store the model
    # the output is a SpatialCluster of an ArimaModelRegion, which has a single ARIMA model in it.
    arima_medoids_numpy = np.empty((x_len, y_len), dtype=object)
    arima_model_cluster = trainer.train_auto_arima_at_medoid(partition, training_cluster,
                                                             arima_medoids_numpy)

    # recover the ArimaModelRegion inside the cluster, this can do forecasting
    arima_model_region = arima_model_cluster.decorated_region

    # this will evaluate the forecast errors
    error_analysis = ErrorAnalysis(observation_cluster, training_region=training_cluster,
                                   parallel_workers=None)

    # use the model at the medoid to create a forecast
    # the model does not require any other parameters, but needs forecast_len set
    arima_model_region.forecast_len = forecast_len
    arima_at_medoid = arima_model_region.function_at(cluster.centroid)
    forecast_series = arima_at_medoid(None)

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
    # csv/<region>/<clustering>/<distance>/<auto_arima>_distance_errors_cluster<i>.csv
    csv_dir = trainer.metadata.csv_dir
    fs_util.mkdir(csv_dir)

    csv_filename = '{!r}_distance_errors_cluster_{}.csv'.format(trainer.metadata.model_params,
                                                                cluster.name)
    csv_full_path = os.path.join(csv_dir, csv_filename)

    # the CSV has this format:
    # <point> <distance(point, medoid)> <forecast error of ARIMA at medoid with point test data>
    with open(csv_full_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        # write header
        distance_str = 'distance_{}'.format(trainer.distance_measure)
        csv_writer.writerow(['position', distance_str, 'forecast_error'])

        # each point in the cluster is a tuple in the CSV
        for i, (point_in_cluster, value) in enumerate(cluster):

            # transform the point to absolute coordinates (from original dataset)
            coords = trainer.region_metadata.absolute_position_of_point(point_in_cluster)

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
            '{}'.format(trainer.clustering_metadata)))

        # save the plot here
        plot_dir = trainer.metadata.plot_dir
        fs_util.mkdir(plot_dir)
        plot_name = '{!r}_distance_errors_cluster_{}.pdf'.format(trainer.metadata.model_params,
                                                                 cluster.name)
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
