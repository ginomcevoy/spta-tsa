'''
Execute this program to:

1. Partition a spatio-temporal region using a clustering algorithm (e.g. k-medoids)
2. Train auto ARIMA models at the medoids of the resulting clusters.
3. Calculate the generalization error of each model when forecasting the test series at all the
   medoids.
4. Save the results as CSV format.

Example: with k=4, there will be 4 medoids, train an auto ARIMA model for each point. Then use
each model to compute the generalization error at all medoids, this will result in 4x4 errors.
'''

import argparse
import csv
import os

from spta.distance.dtw import DistanceByDTW

from spta.region.error import ErrorAnalysis, error_functions
from spta.region.train import SplitTrainingAndTestLast
from spta.solver.auto_arima import AutoARIMATrainer

from spta.util import fs as fs_util
from spta.util import log as log_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.clustering import get_suite
from experiments.metadata.region import predefined_regions


def processRequest():

    desc = '''Given a partitioned spatio-temporal region and a clustering algorithm, train auto
ARIMA models at the medoids of the resulting clusters. Then compute the generalization errors
of each model at all the cluster medoids, resulting in k*k errors.'''

    usage = '%(prog)s [-h] <region> <auto_arima_cluster_id> [--flen <forecast_length>]' \
        '[--error error_type] [--log=log_level]'
    parser = argparse.ArgumentParser(prog='auto-arima-errors', description=desc, usage=usage)

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

    # optional arguments
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_auto_arima_errors(args, logger)


def do_auto_arima_errors(args, logger):

    # parse to get metadata
    region_metadata, clustering_suite, auto_arima_params, forecast_len, error_type = \
        metadata_from_args(args)

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()

    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        do_auto_arima_errors_for_clustering(spt_region, region_metadata, clustering_metadata,
                                            auto_arima_params, forecast_len, error_type, logger)

def do_auto_arima_errors_for_clustering(spt_region, region_metadata, clustering_metadata,
                                        auto_arima_params, forecast_len, error_type, logger):

    # prepare to train a solver based on auto ARIMA
    trainer = AutoARIMATrainer(region_metadata=region_metadata,
                               clustering_metadata=clustering_metadata,
                               distance_measure=DistanceByDTW(),
                               auto_arima_params=auto_arima_params)

    # use the trainer to get the cluster partition and corresponding medoids
    # will try to leverage pickle and load previous attempts, otherwise calculate and save
    trainer.prepare_for_training()
    partition = trainer.clustering_algorithm.partition(spt_region,
                                                       with_medoids=True,
                                                       save_csv_at='outputs',
                                                       pickle_home='pickle')
    medoids = partition.medoids

    # create training/test regions
    splitter = SplitTrainingAndTestLast(forecast_len)
    (training_region, test_region) = splitter.split(spt_region)

    # this will evaluate the forecast errors
    error_analysis = ErrorAnalysis(test_region, training_region=training_region,
                                   parallel_workers=None)

    # use the trainer again to run auto ARIMA at medoids
    arima_model_region = trainer.train_auto_arima_at_medoids(training_region, partition,
                                                             medoids)

    # prepare the CSV output for this clustering partition
    csv_dir = trainer.metadata.output_dir('outputs')
    fs_util.mkdir(csv_dir)

    # csv_filename = 'auto_arima_{}_errors_at_medoids.csv'.format(auto_arima_params)
    csv_filename = 'error-between-medoids__{!r}__{}.csv'.format(clustering_metadata, error_type)
    csv_full_path = os.path.join(csv_dir, csv_filename)

    with open(csv_full_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        # write header
        indices = range(0, clustering_metadata.k)
        header_indices = [
            str(index)
            for index
            in indices
        ]
        header_row = ['cluster'] + header_indices
        csv_writer.writerow(header_row)

    for i, medoid in enumerate(medoids):
        find_auto_arima_errors_for_medoid(spt_region, error_analysis, arima_model_region, i,
                                          medoid, medoids, csv_full_path, forecast_len, error_type,
                                          logger)

    logger.info('Saved CSV to {}'.format(csv_full_path))

def find_auto_arima_errors_for_medoid(spt_region, error_analysis, arima_model_region, i, medoid,
                                      medoids, csv_full_path, forecast_len, error_type, logger):

    # use the model at the medoid to create a forecast
    # the model does not require any other parameters, but needs forecast_len set
    arima_model_region.forecast_len = forecast_len
    arima_at_medoid = arima_model_region.function_at(medoid)
    forecast_series = arima_at_medoid(None)

    # calculate the forecast error in the entire region, then for each medoid
    error_region = error_analysis.with_repeated_forecast(forecast_series, error_type)
    errors_at_medoids = [
        error_region.value_at(other_medoid)
        for other_medoid
        in medoids
    ]

    # format to string
    errors_at_medoids_str = [
        '{:.3f}'.format(error_at_other_medoid)
        for error_at_other_medoid
        in errors_at_medoids
    ]

    log_msg = 'Prediction errors for model at medoid {} in other medoids {} = {}'
    logger.debug(log_msg.format(medoid, medoids, errors_at_medoids))

    # append to CSV
    with open(csv_full_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([str(i)] + errors_at_medoids_str)


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

    forecast_len = args.flen
    error_type = args.error

    return region_metadata, clustering_suite, auto_arima_params, forecast_len, error_type


if __name__ == '__main__':
    processRequest()
