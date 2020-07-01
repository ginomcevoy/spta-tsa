'''
Execute this program with --train to partition a spatio-temporal region using k-medoids, and then
train auto ARIMA models at the medoids of the resulting clusters. This will create a solver for
forecast queries that can be used with this same command (without --train).
'''

import argparse
import numpy as np
import sys

from spta.region import Region
from spta.region.error import error_functions
from spta.kmedoids import kmedoids
from spta.solver import kmedoids_auto_arima
from spta.util import log as log_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.region import predefined_regions


def processRequest():

    # the parent for both train train and predict actions
    # it cannot be the main parser, because it would add the subparsers recursively... (very ugly)
    parent_parser = argparse.ArgumentParser(add_help=False)
    configure_parent(parent_parser)

    # the main parser
    desc = '''Solver: given a partitioned spatio-temporal region and trained auto
ARIMA models at the medoids of the resulting clusters.
Assumes DTW and default k-medoids. Use train to train the solver'''

    usage = '%(prog)s [-h] {train | predict} ...'
    main_parser = argparse.ArgumentParser(prog='auto-arima-solver', description=desc,
                                          usage=usage)

    # create subparsers for train and predict actions
    subparsers = main_parser.add_subparsers(title="actions")

    train_parser = subparsers.add_parser("train", parents=[parent_parser],
                                         # add_help=False,
                                         description="Parse train action",
                                         help="Train and save the solver")
    configure_train_parser(train_parser)

    predict_parser = subparsers.add_parser("predict", parents=[parent_parser],
                                           # add_help=False,
                                           description="Parse predict action",
                                           help="Use a solver to create predict value for region")
    configure_predict_parser(predict_parser)

    # show help if no subcommand is given
    if len(sys.argv) == 1:
        parent_parser.print_help(sys.stderr)
        sys.exit(1)

    # activate parsing and sub-command function call
    # note: we expect args.func(args) to succeed, since we are making sure we have subcommands
    args = main_parser.parse_args()
    args.func(args)


def configure_parent(parent_parser):
    '''
    Common mandatory and optional arguments for both train and predict
    '''

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parent_parser.add_argument('region', help='Name of the region metadata',
                               choices=region_options)

    # auto_arima_id, see metadata.arima
    auto_arima_options = predefined_auto_arima().keys()
    parent_parser.add_argument('auto_arima', help='ID of auto arima clustering experiment',
                               choices=auto_arima_options)

    parent_parser.add_argument('-k', help='Number of clusters for k-medoids', required=True)
    parent_parser.add_argument('-seed', help='Random seed for k-medoids', required=True)

    # error type is optional and defaults to sMAPE
    error_options = error_functions().keys()
    error_help_msg = 'error type (default: %(default)s)'
    parent_parser.add_argument('--error', help=error_help_msg, default='sMAPE',
                               choices=error_options)

    # other optional arguments
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parent_parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)


def configure_train_parser(train_parser):
    '''
    Configure specific options for the predict action.
    '''
    # function called after action is parsed
    train_parser.set_defaults(func=train_request)


def configure_predict_parser(predict_parser):
    '''
    Configure specific options for the predict action.
    '''
    # region is mandatory... as 4 coords for now
    predict_parser.add_argument('x1', help='Region for forecasting: region.x1')
    predict_parser.add_argument('x2', help='Region for forecasting: region.x2')
    predict_parser.add_argument('y1', help='Region for forecasting: region.y1')
    predict_parser.add_argument('y2', help='Region for forecasting: region.y2')

    # function called after action is parsed
    predict_parser.set_defaults(func=predict_request)


def train_request(args):

    logger = log_util.setup_log_argparse(args)
    logger.debug(args)

    # parse to get metadata
    region_metadata, kmedoids_metadata, auto_arima_params = metadata_from_args(args)

    # get a trainer, and train to get a solver
    # default values for test/training...
    trainer = kmedoids_auto_arima.KmedoidsAutoARIMATrainer(region_metadata, kmedoids_metadata,
                                                           auto_arima_params)
    solver = trainer.train(args.error)

    # persist this solver for later use
    solver.save()


def predict_request(args):

    logger = log_util.setup_log_argparse(args)
    logger.debug(args)

    # parse to get metadata
    region_metadata, kmedoids_metadata, auto_arima_params = metadata_from_args(args)

    # get region
    prediction_region = Region(int(args.x1), int(args.x2), int(args.y1), int(args.y2))

    # load solver from persistence
    pickler = kmedoids_auto_arima.KmedoidsAutoARIMASolverPickler(region_metadata,
                                                                 kmedoids_metadata,
                                                                 auto_arima_params,
                                                                 args.error)
    solver = pickler.load_solver()
    print('')
    print('*********************************')
    print('Using solver:')
    print(str(solver))
    print('*********************************')
    print('')

    # for printing forecast and error values
    np.set_printoptions(precision=3)

    # this has all the necessary information and can be iterated
    prediction_result = solver.predict(prediction_region)

    for relative_point in prediction_result:

        coords = prediction_result.absolute_coordinates_of(relative_point)
        cluster_index = prediction_result.cluster_index_of(relative_point)
        forecast = prediction_result.forecast_at(relative_point)
        error = prediction_result.error_at(relative_point)
        print('*********************************')
        print('Point: {} (cluster {})'.format(coords, cluster_index))
        print('Forecast: {}'.format(forecast))
        print('Error ({}): {:.3f}'.format(args.error, error))

    prediction_result.save_as_csv()


def metadata_from_args(args):
    '''
    Metadata common to both train and predict.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # create kmedoids metadata, assuming default values
    k, random_seed = int(args.k), int(args.seed)
    kmedoids_metadata = kmedoids.kmedoids_default_metadata(k, random_seed=random_seed)

    # get the auto arima params
    auto_arima_params = predefined_auto_arima()[args.auto_arima]

    return region_metadata, kmedoids_metadata, auto_arima_params


if __name__ == '__main__':
    processRequest()
