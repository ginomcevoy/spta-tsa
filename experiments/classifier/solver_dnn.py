'''
Execute this program with the train subcommand to partition a spatio-temporal region using a
clustering algorithm (e.g. k-medoids), and then train auto ARIMA models at the medoids of the
resulting clusters.

This will create a solver for forecast queries that can be used with this same command and the
predict subcommand.
'''

import argparse
import numpy as np

from spta.arima import AutoArimaParams
from spta.arima.train import TrainerAutoArima

from spta.region import Region
from spta.distance.dtw import DistanceByDTW
from spta.model.error import error_functions

from spta.classifier.model import SolverFromClassifier

from spta.util import log as log_util

from experiments.metadata.classifier import classifier_experiments
from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.arima_clustering import auto_arima_clustering_experiments
from experiments.metadata.clustering import get_suite as get_clustering_suite
from experiments.metadata.region import predefined_regions

# default number of samples used for testing
TEST_SAMPLES = 8


def call_parser():

    desc = '''DNN Solver: Given a prediction region, use a DNN classifier to retrieve, for each point P,
the most appropriate medoid. Use the auto ARIMA model at medoid M to create a forecast for point P.'''

    usage = '%(prog)s <region_id> <classifier_id> <lat1 lat2 long1 long2> [--error error_type ' \
        '[--tf forecast_len] [--log log_level]'
    parser = argparse.ArgumentParser(prog='classifier-solver', description=desc, usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # auto_arima_id, see metadata.arima_clustering
    auto_arima_clustering_options = auto_arima_clustering_experiments().keys()
    parser.add_argument('auto_arima_clustering_id', help='ID of auto arima clustering experiment',
                        choices=auto_arima_clustering_options)

    # classifier_id, see metadata.classifier
    classifier_options = classifier_experiments().keys()
    parser.add_argument('classifier', help='ID of classifier',
                        choices=classifier_options)

    # region is mandatory... as 4 coords for now
    parser.add_argument('lat1', help='First latitude index relative to region (north)')
    parser.add_argument('lat2', help='Second latitude index relative to region (south)')
    parser.add_argument('long1', help='First longitude index relative to region (west)')
    parser.add_argument('long2', help='Second longitude index relative to region (east)')

    # error type is optional and defaults to sMAPE
    error_options = error_functions().keys()
    error_help_msg = 'error type (default: %(default)s)'
    parser.add_argument('--error', help=error_help_msg, default='sMAPE', choices=error_options)

    # tf (forecast length) is optional and defaults to 8
    forecast_help_msg = 'number of samples for forecast/testing (default: %(default)s)'
    parser.add_argument('--tf', help=forecast_help_msg, default=8, type=int)

    # other optional arguments
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()
    process_request(args)

def process_request(args):

    logger = log_util.setup_log_argparse(args)
    logger.debug(args)

    # parse to get metadata
    region_metadata, clustering_suite, auto_arima_params, classifier_params, error_type = metadata_from_args(args)
    forecast_len = args.tf

    distance_measure = DistanceByDTW()

    # get region
    # NOTE: x = lat, y = long!
    # Revise this if we adapt for COORDS
    prediction_region = Region(int(args.lat1), int(args.lat2), int(args.long1), int(args.long2))

    # TODO get rid of trainer? it is only used to create ArimaModelRegion...
    # TODO get rid of clustering suite? apparently no effective use for it
    model_trainer = TrainerAutoArima(auto_arima_params, region_metadata.x_len, region_metadata.y_len)
    solver = SolverFromClassifier(region_metadata=region_metadata,
                                  distance_measure=distance_measure,
                                  clustering_suite=clustering_suite,
                                  model_trainer=model_trainer,
                                  model_params=auto_arima_params,
                                  classifier_params=classifier_params,
                                  test_len=forecast_len,
                                  error_type=error_type)

    prediction_result = solver.predict(prediction_region, tp=classifier_params.window_size, output_home='outputs')

    # for printing forecast and error values
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

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

    # get the clustering metadata from auto-ARIMA clustering experiment
    experiment = auto_arima_clustering_experiments()[args.auto_arima_clustering_id]

    clustering_suite = get_clustering_suite(experiment.clustering_type,
                                            experiment.clustering_suite_id)

    auto_arima_params = predefined_auto_arima()[experiment.auto_arima_id]
    classifier_params = classifier_experiments()[args.classifier]

    return region_metadata, clustering_suite, auto_arima_params, classifier_params, args.error


if __name__ == '__main__':
    call_parser()
