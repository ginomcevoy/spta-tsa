'''
Execute this program with the train subcommand to partition a spatio-temporal region using a
clustering algorithm (e.g. k-medoids), and then, for each point in the region, k-NN algorithm
to predict the future values of the series at each point.

This will create a baseline solver to be compared with more sophisticated approaches.
'''

import argparse
import numpy as np

from spta.clustering.kmedoids import KmedoidsClusteringMetadata
from spta.clustering.regular import RegularClusteringMetadata

from spta.distance.dtw import DistanceByDTW

from spta.model.forecast import ForecastAnalysis
from spta.model.knn import TrainerKNN, KNNParams
from spta.model.error import error_functions

from spta.region import Region
from spta.region.scaling import SpatioTemporalScaled

from spta.solver.model import SolverPickler
from spta.solver.train import SolverTrainer
from spta.solver.metadata import SolverMetadataBuilder
from spta.solver.result import PredictionQueryResultBuilder

from spta.util import log as log_util

from experiments.metadata.region import predefined_regions


# default number of samples used for testing
TEST_SAMPLES = 8


def call_parser():

    desc = '''KNNSolver: answer a forecast query using k-NN on each point of a prediction region, in order to make
an in-sample prediction of the series, then calculate the forecast errors on each point.
Finally, print the results and save as CSV.'''

    usage = '%(prog)s <region_id> <k> <lat1 lat2 long1 long2> [--error error_type ' \
        '[--tf forecast_len] [--log log_level]'
    parser = argparse.ArgumentParser(prog='knn-solver-each', description=desc, usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    parser.add_argument('k', help='Value of k for k-NN', type=int)

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
    parser.add_argument('--tf', help=forecast_help_msg, default=TEST_SAMPLES, type=int)

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
    region_metadata, knn_model_params, error_type = metadata_from_args(args)
    forecast_len = args.tf

    # get region
    # NOTE: x = lat, y = long!
    # Revise this if we adapt for COORDS
    prediction_region = Region(int(args.lat1), int(args.lat2), int(args.long1), int(args.long2))

    # create metadata without clustering support
    # the number of test samples used to train the models is equal to the forecast length,
    # to keep everything 'in-sample'
    metadata_builder = SolverMetadataBuilder(region_metadata=region_metadata,
                                             model_params=knn_model_params,
                                             test_len=forecast_len,
                                             error_type=error_type)
    solver_metadata = metadata_builder.build()

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()

    # subset to get prediction region
    prediction_spt_region = spt_region.region_subset(prediction_region)

    # get a trainer using metadata
    # the training requires a number of samples used as test data (data retained from the model
    # in order to calculate forecast error), this is test_len (--test)
    knn_model_trainer = TrainerKNN(knn_model_params, prediction_spt_region.x_len, prediction_spt_region.y_len)
    forecast_analysis = ForecastAnalysis(knn_model_trainer, parallel_workers=None)
    forecast_analysis.train_models(prediction_spt_region, test_len=forecast_len)

    # do in-sample forecasting with models at each point, evaluate error
    forecast_region_each, error_region_each, time_forecast = \
        forecast_analysis.forecast_at_each_point(forecast_len, error_type)

    # also need the test subregion for results
    test_subregion = forecast_analysis.test_region

    # handle descaling here: we want to present descaled data to users
    if region_metadata.scaled:

        forecast_region_each = descale_subregion(forecast_region_each, prediction_spt_region,
                                                 logger)
        test_subregion = descale_subregion(test_subregion, prediction_spt_region, logger)

    # prepare the results with the data gathered
    # is_future is False, we always use in-sample forecasting here
    result_builder = PredictionQueryResultBuilder(solver_metadata=solver_metadata,
                                                  forecast_len=forecast_len,
                                                  forecast_subregion=forecast_region_each,
                                                  test_subregion=test_subregion,
                                                  error_subregion=error_region_each,
                                                  prediction_region=prediction_region,
                                                  spt_region=spt_region,
                                                  output_home='outputs',
                                                  is_out_of_sample=False)
    prediction_result = result_builder.build()

    # for printing forecast and error values
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # make a prediction, the forecast may be in-sample (future=False) or out-of-sample
    # (future=True)
    # The prediction result has all the necessary information and can be iterated by point
    # prediction_result = solver.predict(prediction_region, is_future=args.future)

    for relative_point in prediction_result:

        # for each point, the result can print a text output
        print('*********************************')
        text = prediction_result.lines_for_point(relative_point)
        print('\n'.join(text))

    # the result can save relevant information to CSV
    prediction_result.save_as_csv()


def descale_subregion(subregion, prediction_spt_region, logger):
    '''
    The forecast_region_each and test_subregion are not aware of the scaling.
    As a workaround, use the spatio-temporal region of the prediction subset
    (which HAS the scaling data) and to retrieve appropriate descaling info.
    '''
    logger.debug('About to descale manually: {}'.format(subregion))
    subregion_with_scaling = SpatioTemporalScaled(subregion,
                                                  scale_min=prediction_spt_region.scale_min,
                                                  scale_max=prediction_spt_region.scale_max)
    return subregion_with_scaling.descale()


def metadata_from_args(args):
    '''
    Metadata common to both train and predict.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # the k-NN params is the value of k and the distance measure
    knn_model_params = KNNParams(args.k, DistanceByDTW())

    return region_metadata, knn_model_params, args.error


if __name__ == '__main__':
    call_parser()
