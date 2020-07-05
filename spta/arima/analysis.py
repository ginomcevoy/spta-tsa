from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time

from spta.util import log as log_util

from . import ArimaPDQ, AutoArimaParams

# default forecast length
FORECAST_LENGTH = 8


ArimaErrors = namedtuple('ArimaErrors', ('each', 'minimum', 'min_local', 'centroid', 'maximum'))


class ArimaErrorAnalysis(log_util.LoggerMixin):
    '''
    Perform the following analysis on a spatio-temporal region:

    1. Build one ARIMA model for each point, forecast FORECAST_LENGTH days and compute the error
       of each forecast. This will create an error region (error_region_each).
       Combine the errors using distance_measure.combine to obtain a single metric for the
       prediction error. (error_region_each.combined_error)

    2. Consider ARIMA models at different points as representatives of the region. For a given
       point, use the forecast made by the model at *that* point, and use it to predict the
       observed values in the entire region. Obtain the MASE errors for each point
       ("error_single_model"), then compute the overall error made by combining the errors (RMSE).

    3. Consider the following points for 2.:
        - minimum: the point that minimizes the error_single_model, i.e. has the minimum RMSE
            of the MASE errors on each point, when using its ARIMA model to forecast the
            entire region.

        - min_local: the point that has the smallest forecast MASE error when forecasting its own
            observation.

        - centroid: the centroid of the region calculated externally, or using DTW if not provided.

        - maximum: the point that maximizes its error_single_model. This is the worst possible
            ARIMA model for the region, used for comparison.

    The centroid should be provided in the spatio-temporal region as spt_region.centroid
    (it does not depend on ARIMA, only on the dataset). If not provided, error is np.nan.
    '''

    def __init__(self, arima_forecasting):

        # delegate tasks to this implementation
        self.arima_forecasting = arima_forecasting

    def evaluate_forecast_errors(self, spt_region, error_type, forecast_len=FORECAST_LENGTH):
        '''
        Performs the main analysis, logs progress and returns the results.
        See spta.region.error.get_error_func for available error types.
        '''

        # orchestrate the tasks of ArimaForecasting to achieve the requested results
        # time everything, this will be the compute time
        time_start = time.time()

        # train
        self.arima_forecasting.train_models(spt_region, forecast_len)

        # forecast using ARIMA model at each point
        each_result = self.arima_forecasting.forecast_at_each_point(forecast_len, error_type)
        (_, error_region_each, forecast_time) = each_result

        overall_error_each = error_region_each.overall_error
        log_msg = 'Combined {} error from all ARIMAs: {}'
        self.logger.info(log_msg.format(error_type, overall_error_each))

        if spt_region.has_centroid:
            # show the local error at the centroid
            centroid = spt_region.centroid
            local_error_centroid = error_region_each.value_at(centroid)
            log_msg = 'Local error of ARIMA model at the medoid {}: {}'
            self.logger.debug(log_msg.format(centroid, local_error_centroid))

        # find the errors when using each model to forecast the entire region
        # see ArimaForecasting for details
        overall_error_region = self.arima_forecasting.forecast_whole_region_with_all_models(
            forecast_len, error_type)

        # forecast computations finish here
        time_end = time.time()
        compute_time = time_end - time_start

        #
        # Best ARIMA model: the one that minimizes the overall error when it is used to forecast
        # the entire region
        #
        (point_overall_error_min, overall_error_min) = overall_error_region.find_minimum()
        log_msg = 'Minimum overall {} error with single ARIMA at {}: {}'
        self.logger.info(log_msg.format(error_type, point_overall_error_min, overall_error_min))

        #
        # ARIMA using the model with minimum local error
        # This is the model that yields the minimum error when forecasting its own series.
        # The error computed is the error when using *that* model to forecast the entire region.
        #
        (point_min_local_error, _) = error_region_each.find_minimum()
        overall_error_min_local = overall_error_region.value_at(point_min_local_error)

        log_msg = '{} error from ARIMA model with min local error at {}: {}'
        self.logger.info(log_msg.format(error_type, point_min_local_error,
                                        overall_error_min_local))

        #
        # ARIMA using the model at the centroid
        # ask the region for its centroid. If not available, the error is np.nan
        #
        if spt_region.has_centroid:
            centroid = spt_region.centroid
            overall_error_centroid = overall_error_region.value_at(centroid)
        else:
            centroid = None
            overall_error_centroid = np.nan
            self.logger.warn('Centroid was not pre-calculated!')

        log_msg = '{} error from ARIMA model at centroid {}: {}'
        self.logger.info(log_msg.format(error_type, centroid, overall_error_centroid))

        #
        # Worst ARIMA model: the one that maximizes the overall error when it is used to
        # to forecast the entire region
        #
        (point_overall_error_max, overall_error_max) = overall_error_region.find_maximum()
        log_msg = 'Maximum overall {} error with single ARIMA at {}: {}'
        self.logger.info(log_msg.format(error_type, point_overall_error_max, overall_error_max))

        # gather all errors
        overall_errors = ArimaErrors(overall_error_each, overall_error_min,
                                     overall_error_min_local, overall_error_centroid,
                                     overall_error_max)

        # results, the arima_forecasting object contains data about regions and forecasts
        return self.arima_forecasting, overall_errors, forecast_time, compute_time

    def plot_one_arima(self, point):

        training_region = self.arima_forecasting.training_region
        forecast_region = self.arima_forecasting.forecast_region_each
        test_region = self.arima_forecasting.test_region
        arima_region = self.arima_forecasting.arima_models

        train_point = training_region.series_at(point)
        forecast_point = forecast_region.series_at(point)
        test_point = test_region.series_at(point)

        (training_len, _, _) = training_region.shape
        (test_len, _, _) = test_region.shape
        test_index = np.arange(training_len, training_len + test_len)

        train_series = pd.Series(train_point)
        forecast_series = pd.Series(forecast_point, index=test_index)
        test_series = pd.Series(test_point, index=test_index)
        model_point = arima_region.value_at(point)
        print(model_point.__class__.__name__)
        print(dir(model_point))

        # FIXME arima_model.ArimaResults vs arima.model.ArimaResults
        model_point.plot_predict(dynamic=False)
        plt.show()

        # Plot
        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(train_series, label='training')
        plt.plot(test_series, label='actual')
        plt.plot(forecast_series, label='forecast')
        # plt.fill_between(lower_series.index, lower_series, upper_series,
        #                  color='k', alpha=.15)
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()


if __name__ == '__main__':

    log_util.setup_log('DEBUG')

    # get region from metadata
    from spta.region import Point, Region
    from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata
    from .forecast import ArimaForecastingPDQ, ArimaForecastingAutoArima

    nordeste_small_md = SpatioTemporalRegionMetadata(
        'nordeste_small', Region(43, 50, 85, 95), series_len=365, ppd=1, last=True)
    spt_region = SpatioTemporalRegion.from_metadata(nordeste_small_md)

    # region has known centroid
    spt_region.centroid = Point(5, 4)

    # use these parameters for ARIMA analysis
    arima_params = ArimaPDQ(1, 1, 1)
    forecast_len = 8
    parallel_workers = 4

    forecasting_pdq = ArimaForecastingPDQ(arima_params, parallel_workers)
    analysis_pdq = ArimaErrorAnalysis(forecasting_pdq)
    analysis_pdq.evaluate_forecast_errors(spt_region, 'MASE', forecast_len)

    # check forecast at centroid
    # FIXME plot_predict does not work with pmdarima or new arima.model.ARIMA
    # analysis_pdq.plot_one_arima(Point(5, 4))

    auto_arima_params = AutoArimaParams(1, 1, 3, 3, 1, True)
    forecasting_auto = ArimaForecastingAutoArima(auto_arima_params, parallel_workers)
    analysis_auto = ArimaErrorAnalysis(forecasting_auto)
    analysis_auto.evaluate_forecast_errors(spt_region, 'MASE', forecast_len)

    # check forecast at centroid
    # FIXME plot_predict does not work with pmdarima or new arima.model.ARIMA
    # analysis_auto.plot_one_arima(Point(5, 4))
