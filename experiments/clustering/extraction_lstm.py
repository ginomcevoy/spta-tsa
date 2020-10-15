'''
TODO doco
'''

import argparse

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.region import predefined_regions
from experiments.metadata.clustering import get_suite, suite_options

from spta.region.error import error_functions
from spta.classifier.train_input import TrainDataWithRandomPoints, MedoidSeriesFormatter, CHOICE_CRITERIA
from spta.distance.dtw import DistanceByDTW

from spta.util import log as log_util


TEST_SAMPLES = 8

def processRequest():

    # parses the arguments
    desc = 'Extraction LSTM...'

    usage = '%(prog)s [-h] <region> [kmedoids|regular] <clustering_suite> <criterion> [--random=1000] ' \
        '[--random-seed=0] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-extraction-lstm', description=desc,
                                     usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # required argument: clustering algorithm
    clustering_options = ('kmedoids', 'regular')
    parser.add_argument('clustering_type', help='Type of clustering algorithm',
                        choices=clustering_options)

    # required argument: clustering ID
    parser.add_argument('clustering_suite', help='ID of the clustering suite',
                        choices=suite_options())

    # required argument: criterion to choose medoid
    parser.add_argument('criterion', help='How to choose the medoid for a random point', choices=CHOICE_CRITERIA)

    # random parameters
    help_msg = 'Number of random points for extraction (default: %(default)s)'
    parser.add_argument('--random', help=help_msg, default=1000, type=int)

    help_msg = 'Random seed for extraction (default: %(default)s)'
    parser.add_argument('--random-seed', help=help_msg, default=0, type=int)

    # logging
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    # here we have optional arguments that are only required when using criterion = 'min_error'

    # auto_arima_id, see metadata.arima
    auto_arima_options = predefined_auto_arima().keys()
    parser.add_argument('--auto-arima', help='ID of auto arima experiment', default='simple',
                        choices=auto_arima_options)

    # tp stands for "time past"
    # The number of samples used as test series (out-of-sample test) to decide how to train a model
    # (training subset) and calculate the forecast error (test_subset)
    # the value used in predict needs to match a previously trained solver
    help_msg = 'Number of past samples for testing (default: %(default)s)'
    parser.add_argument('--tp', help=help_msg, default=TEST_SAMPLES, type=int)

    # error type is optional and defaults to sMAPE
    error_options = error_functions().keys()
    error_help_msg = 'error type (default: %(default)s)'
    parser.add_argument('--error', help=error_help_msg, default='sMAPE', choices=error_options)

    args = parser.parse_args()

    logger = log_util.setup_log_argparse(args)
    analyze_suite(args, logger)


def analyze_suite(args, logger):

    region_metadata, clustering_suite = metadata_from_args(args)

    # default...
    output_home = 'outputs'

    # TODO assuming DTW
    distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    # TODO this code is broken if we don't use DTW
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    choice_args = solver_metadata_from_args(args)

    # this will find the medoid that minimizes its distance with list of random points
    train_data_generator = TrainDataWithRandomPoints(region_metadata, distance_measure)
    train_data_generator.evaluate_score_of_random_points(clustering_suite=clustering_suite,
                                                         count=args.random,
                                                         random_seed=args.random_seed,
                                                         criterion=args.criterion,
                                                         output_home=output_home,
                                                         **choice_args)

    # output medoid data as CSV
    medoid_formatter = MedoidSeriesFormatter(region_metadata, distance_measure, clustering_suite)
    medoid_formatter.produce_csv(output_home)


def metadata_from_args(args):
    '''
    Extract experiment metadata from request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata
    # FIXME need to change metadata.clustering so that the identifier is passed to the suite
    # here we pass it manually
    clustering_suite = get_suite(args.clustering_type, args.clustering_suite)
    clustering_suite.identifier = args.clustering_suite

    # # a meaningful description to use in the CSV name
    # suite_desc = '{}-{}'.format(args.clustering_type, args.clustering_suite)

    return region_metadata, clustering_suite  # , suite_desc


def solver_metadata_from_args(args):
    '''
    When using solvers to choose a medoid, the proper metadata is needed.
    '''
    choice_args = {
        'model_params': predefined_auto_arima()[args.auto_arima],  # assuming auto ARIMA
        'test_len': args.tp,
        'error_type': args.error,
    }

    return choice_args


if __name__ == '__main__':
    processRequest()
