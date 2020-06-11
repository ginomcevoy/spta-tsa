'''
Forecasting with ARIMA models. Contains:

- ArimaModelRegion, a function region that holds a trained ARIMA model at each point. Can create
    a forecast region when called with an empty region.

- ArimaForecasting, a class that orchestrates the training and forecasting of ARIMA models.
'''
import numpy as np
import time

from spta.region.function import FunctionRegionSeries
from spta.region.train import SplitTrainingAndTestLast
from spta.region.error import ErrorAnalysis

from spta.util import log as log_util

from .training import ArimaTrainer

# types of forecasting errors supported
FORECAST_ERROR_TYPES = ['MASE']


class ArimaModelRegion(FunctionRegionSeries):
    '''
    A FunctionRegion that uses the arima_forecast function to create a forecast region using
    ARIMA models. This means that applying this FunctionRegion to a region with another region,
    the result will be a spatio-temporal region, where each point (i, j) is a forecast series of
    the model at (i, j).

    The instance of this class already has a trained model, and already has the forecast length.
    This means that the region to which it is applied does not need any particular information.
    It can be an empty SpatialRegion, the value at each point will not be used. The points of the
    SpatialRegion are still used for iteration.
    '''

    def function_at(self, point):
        '''
        Override the method that returns the function at each point (i, j). This is needed because
        this function region does not really store a function, it stores a model object.

        We want to have the model object to inspect some properties, the model is still
        retrievable using value_at(point).
        '''
        # the region stores the models at (i, j), extract it and return the forecast result.
        model_at_point = self.value_at(point)

        # wrap the call to ARIMA forecast, to match the signature expected by the input region.

        # TODO this could be refactored to a wrapping model class, making the code more
        # generic when SVM comes! (a wrapper subclass for ARIMA, a wrapper subclass for SVM)
        def forecast_from_model(value):
            # ignore the value of the input region, we already have forecast_len available
            if model_at_point is None:
                # no model, return array of NaNs
                return np.repeat(np.nan, repeats=self.output_len)
            else:
                # return a forecast array, [0] because the forecast function returns an array
                # with the forecast array at index 0.
                return model_at_point.forecast(self.output_len)[0]

        return forecast_from_model


class ArimaForecasting(log_util.LoggerMixin):
    '''
    Manages the training and forecasting of ARIMA models.
    '''

    def __init__(self, arima_params, forecast_len, parallel_workers=None):
        self.arima_params = arima_params
        self.forecast_len = forecast_len
        self.parallel_workers = parallel_workers

        # created when training
        self.arima_models = None
        self.error_analysis = None

        # created when calling forecast_at_each_point
        self.forecast_region_each = None

    def train_models(self, spt_region):
        '''
        Separate a spatio-temporal region in training region and observation region.
        Then, build one ARIMA model for each point, using its training series.

        Returns a spatial region that contains, for each point an ARIMA model that can be
        later used for forecasting.
        '''

        self.logger.info('Using (p, d, q) = {}'.format(self.arima_params))

        # create training/test regions, where the test region has the same series length as the
        # expected forecast.
        splitter = SplitTrainingAndTestLast(self.forecast_len)
        (self.training_region, self.test_region) = splitter.split(spt_region)

        # a function region with produces trained models when applied to a training region
        arima_trainers = ArimaTrainer.from_training_region(self.training_region, self.arima_params,
                                                           self.forecast_len)

        # train the models: this returns an instance of ArimaModelRegion, that has an instance of
        # statsmodels.tsa.arima.ARIMAResults at each point
        arima_models = arima_trainers.apply_to(self.training_region)

        # save the number of failed models... ugly but works
        arima_models.missing_count = arima_trainers.missing_count

        # save models in instance for later forecasting
        self.arima_models = arima_models

        # prepare the error analysis
        self.error_analysis = ErrorAnalysis(self.test_region, self.training_region,
                                            self.parallel_workers)

        return arima_models

    def forecast_at_each_point(self, error_type):
        '''
        Create a forecast region, using the trained ARIMA model at each point to forecast the
        series at that point. Also, compute the forecast error using the test data as observation.

        Requires a string indicating the type of forecast error to be used.
        See spta.region.error.get_error_func for available error functions.

        Returns the forecast region, the error region and the time it took to compute the forecast.
        '''
        # check conditions
        self.check_forecast_request()

        # prepare a forecast: create an empty region.
        # This region will control the iteration though, and it will be of the same subclass
        # as the training region (and of the spatio-temporal region), thereby supporting clusters.
        empty_region_2d = self.training_region.empty_region_2d()

        # use the ARIMA models to forecast for their respective points, measure time
        time_forecast_start = time.time()
        forecast_region_each = self.arima_models.apply_to(empty_region_2d)
        forecast_region_each.name = 'forecast_region_each'
        time_forecast_end = time.time()
        time_forecast = time_forecast_end - time_forecast_start

        # save the forecast region with each model
        self.forecast_region_each = forecast_region_each

        # calculate the error for this forecast region
        error_region_each = self.error_analysis.with_forecast_region(forecast_region_each,
                                                                     error_type)

        return forecast_region_each, error_region_each, time_forecast

    def forecast_whole_region_with_single_model(self, point, error_type):
        '''
        Using the ARIMA model that was trained at the specified point, forecast the entire region,
        and calculate the error.

        The forecast is the same for each point, since only depends on how the particular model
        was trained.

        Example: create the forecast and evaluate the error using the ARIMA model that was trained
        at the cetroid, and then replicate that forecast over the entire region, effectively using
        a single model for the entire region.

        See spta.region.error.get_error_func for available error functions.

        Returns the error region.
        '''
        # check conditions
        self.check_forecast_request()

        # reuse the forecast at each point
        # if not available, calculate it now
        if self.forecast_region_each is None:
            self.forecast_at_each_point(error_type)

        # now we should have the forecast region
        # use the forecast series at the specified point, and do corresponding error analyis
        assert self.forecast_region_each is not None
        forecast_series = self.forecast_region_each.series_at(point)

        error_region_with_repeated_forecast = \
            self.error_analysis.with_repeated_forecast(forecast_series, error_type)

        return error_region_with_repeated_forecast

    def forecast_whole_region_with_all_models(self, error_type):
        '''
        Consider ARIMA models at different points as representatives of the region. For a given
        point, use the forecast made by the model at *that* point, and use it to predict the
        observed values in the entire region. Obtain the MASE errors for each point, then compute
        the overall error made by combining the errors (RMSE).

        See see spta.region.error.OverallErrorForEachForecast for details.
        See spta.region.error.get_overall_error_func for available error functions.
        '''
        # check conditions
        self.check_forecast_request()

        # reuse the forecast at each point
        # if not available, calculate it now
        if self.forecast_region_each is None:
            self.forecast_at_each_point(error_type)

        # now we should have the forecast region
        assert self.forecast_region_each is not None

        # call appropriate error analysis
        overall_error_region = \
            self.error_analysis.overall_with_each_forecast(self.forecast_region_each, error_type)

        return overall_error_region

    def check_forecast_request(self):
        '''
        Sanity checking for any forecast. Checks that the models have been trained, and that
        the error analysis is available
        '''
        if self.arima_models is None:
            raise ValueError('Forecast requested but models not trained!')

        if self.error_analysis is None:
            raise ValueError('ErrorAnalysis not available!')
