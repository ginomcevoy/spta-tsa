'''
Execute this program with the train subcommand to partition a spatio-temporal region using a
clustering algorithm (e.g. k-medoids), and then train auto ARIMA models at the medoids of the
resulting clusters.

This will create a solver for forecast queries that can be used with this same command and the
predict subcommand.
'''

import argparse
import numpy as np
import sys

from spta.clustering.kmedoids import KmedoidsClusteringMetadata
from spta.clustering.regular import RegularClusteringMetadata

from spta.distance.dtw import DistanceByDTW

from spta.region import Region
from spta.region.error import error_functions
from spta.solver.auto_arima import AutoARIMATrainer, AutoARIMASolverPickler
from spta.solver.metadata import SolverMetadataBuilder
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
Assumes DTW. Use train sub-command to train the solver, and predict to answer prediction queries'''

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

    # clustering algorithm
    clustering_options = ('kmedoids', 'regular')
    parent_parser.add_argument('clustering', help='Name of clustering algorithm',
                               choices=clustering_options)
    parent_parser.add_argument('-k', help='Number of clusters', required=True,
                               type=int)

    # seed of clustering algorithm is optional and defaults to 1
    parent_parser.add_argument('--seed', help='Random seed for k-medoids (default: %(default)s)',
                               default=1, type=int)

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
    predict_parser.add_argument('lat1', help='First latitude index relative to region (north)')
    predict_parser.add_argument('lat2', help='Second latitude index relative to region (south)')
    predict_parser.add_argument('long1', help='First longitude index relative to region (west)')
    predict_parser.add_argument('long2', help='Second longitude index relative to region (east)')

    # optionally request out-of-sample predictions, instead of in-sample
    predict_parser.add_argument('--future', help='predict a future series (out-of-sample) instead',
                                default=False, action='store_true')

    # function called after action is parsed
    predict_parser.set_defaults(func=predict_request)


def train_request(args):

    logger = log_util.setup_log_argparse(args)
    logger.debug(args)

    # parse to get metadata
    region_metadata, clustering_metadata, auto_arima_params = metadata_from_args(args)

    # get a trainer, and train to get a solver
    # default values for test/training...
    trainer = AutoARIMATrainer(region_metadata=region_metadata,
                               clustering_metadata=clustering_metadata,
                               distance_measure=DistanceByDTW(),
                               auto_arima_params=auto_arima_params,
                               error_type=args.error)
    solver = trainer.train()

    # persist this solver for later use
    solver.save()


def predict_request(args):

    logger = log_util.setup_log_argparse(args)
    logger.debug(args)

    # parse to get metadata, assuming DTW
    region_metadata, clustering_metadata, auto_arima_params = metadata_from_args(args)
    distance_measure = DistanceByDTW()

    # get region
    # NOTE: x = lat, y = long!
    # Revise this if we adapt for COORDS
    prediction_region = Region(int(args.lat1), int(args.lat2), int(args.long1), int(args.long2))

    # create metadata with clustering support
    builder = SolverMetadataBuilder(region_metadata=region_metadata,
                                    model_params=auto_arima_params,
                                    error_type=args.error)
    solver_metadata = builder.with_clustering(clustering_metadata=clustering_metadata,
                                              distance_measure=distance_measure).build()

    # load solver from persistence
    pickler = AutoARIMASolverPickler(solver_metadata=solver_metadata)
    solver = pickler.load_solver()
    print('')
    print('*********************************')
    print('Using solver:')
    print(str(solver))
    print('*********************************')
    print('')

    # for printing forecast and error values
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # make a prediction, the forecast may be in-sample (future=False) or out-of-sample
    # (future=True)
    # The prediction result has all the necessary information and can be iterated by point
    prediction_result = solver.predict(prediction_region, is_future=args.future)

    for relative_point in prediction_result:

        # for each point, the result can print a text output
        print('*********************************')
        text = prediction_result.lines_for_point(relative_point)
        print('\n'.join(text))

    # the result can save relevant information to CSV
    prediction_result.save_as_csv()


def metadata_from_args(args):
    '''
    Metadata common to both train and predict.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # create clustering metadata
    if args.clustering == 'kmedoids':
        # assuming other default values for k-medoids
        clustering_metadata = KmedoidsClusteringMetadata(args.k, random_seed=args.seed)

    if args.clustering == 'regular':
        clustering_metadata = RegularClusteringMetadata(args.k)

    # get the auto arima params
    auto_arima_params = predefined_auto_arima()[args.auto_arima]

    return region_metadata, clustering_metadata, auto_arima_params


if __name__ == '__main__':
    processRequest()
