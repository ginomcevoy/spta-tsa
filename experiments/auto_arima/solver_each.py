'''
Execute this program to train auto ARIMA models on each point of a prediction region. Then, for
each point, use its own ARIMA model to make an in-sample prediction of the series, and calculate
the forecast errors for each point. Finally, print the results and save as CSV.

This exhaustive approach (100 models for a 10x10 region) is meant to be compared against the
approach where the representative models are used (at the medoids of each cluster).

TODO Create a proper 'Solver' subclass for this case, need to define a proper interface!
'''

import argparse
import numpy as np

from spta.arima.forecast import ArimaForecastingAutoArima

from spta.region import Region
from spta.model.error import error_functions
from spta.region.scaling import SpatioTemporalScaled

from spta.solver.metadata import SolverMetadataBuilder
from spta.solver.result import PredictionQueryResultBuilder

from spta.util import log as log_util

from experiments.metadata.arima import predefined_auto_arima
from experiments.metadata.region import predefined_regions


def call_parser():

    # configure parser
    desc = '''Execute this program to train auto ARIMA models on each point of a prediction
region. Then, for each point, use its own ARIMA model to make an in-sample prediction of the
series, and calculate the forecast errors for each point. Finally, print the results and save as
CSV.'''
    usage = '%(prog)s <region_id> <auto_arima_id> <lat1 lat2 long1 long2> [--error error_type ' \
        '[--tf forecast_len] [--log log_level]'
    parser = argparse.ArgumentParser(prog='auto-arima-solver-each', description=desc, usage=usage)

    # region_id required, see metadata.region
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # auto_arima_id, see metadata.arima
    auto_arima_options = predefined_auto_arima().keys()
    parser.add_argument('auto_arima', help='ID of auto arima experiment',
                        choices=auto_arima_options)

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
    region_metadata, auto_arima_params, error_type = metadata_from_args(args)
    forecast_len = args.tf

    # get region
    # NOTE: x = lat, y = long!
    # Revise this if we adapt for COORDS
    prediction_region = Region(int(args.lat1), int(args.lat2), int(args.long1), int(args.long2))

    # create metadata without clustering support
    # the number of test samples used to train the models is equal to the forecast length,
    # to keep everything 'in-sample'
    metadata_builder = SolverMetadataBuilder(region_metadata=region_metadata,
                                             model_params=auto_arima_params,
                                             test_len=forecast_len,
                                             error_type=error_type)
    solver_metadata = metadata_builder.build()

    # recover the spatio-temporal region
    spt_region = region_metadata.create_instance()

    # subset to get prediction region
    prediction_spt_region = spt_region.region_subset(prediction_region)

    # delegate model training to this implementation
    # no need for the actual models
    arima_forecasting = ArimaForecastingAutoArima(auto_arima_params)
    arima_forecasting.train_models(prediction_spt_region, test_len=forecast_len)

    # do in-sample forecasting with models at each point, evaluate error
    forecast_region_each, error_region_each, time_forecast = \
        arima_forecasting.forecast_at_each_point(forecast_len, error_type)

    # also need the test subregion for results
    test_subregion = arima_forecasting.test_region

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

    # get the auto arima params
    auto_arima_params = predefined_auto_arima()[args.auto_arima]

    return region_metadata, auto_arima_params, args.error


if __name__ == '__main__':
    call_parser()
