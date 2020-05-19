import numpy as np
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

from spta.distance.dtw import DistanceByDTW
from spta.region import Point, train
from spta.region.function import FunctionRegionScalar, FunctionRegionSeries
from spta.region.forecast import ErrorRegionMASE

from spta.util import arrays as arrays_util
from spta.util import log as log_util

from . import ArimaParams

# default number of data points to forecast and test
FORECAST_LENGTH = 8


class FailedArima(object):
    '''
    Replace failed ARIMA training with an instance of this class.
    It will return a series with NaN when asked to create a forecast.
    '''
    def forecast(self, forecast_len):
        # forecast_series = arima_model.forecast(forecast_len)[0]
        return [np.repeat(np.nan, repeats=forecast_len)]


def train_arima(arima_params, time_series):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.

    Returns a trained ARIMA model than can be used for forecasting (model fit).
    If the evaluation fails, return None instead of the model fit.
    '''
    log = log_util.logger_for_me(train_arima)
    log.debug('Training ARIMA with: %s' % str(arima_params))

    try:
        model = ARIMA(time_series, order=(arima_params.p, arima_params.d, arima_params.q))
        model_fit = model.fit(disp=0)
    except ValueError:
        # log.warn('could not train ARIMA %s for point %s' % (str(arima_params), point))
        # could not train the model
        model_fit = FailedArima()

    return model_fit


class ArimaTrainingRegion(FunctionRegionScalar):
    '''
    A FunctionRegion that uses the train_arima function to train an ARIMA model over a training
    region. Applying this FunctionRegion to the training spatio-temporal region will produce an
    instance of ArimaModelRegion (which is also a SpatialRegion). The shape of ArimaTrainingRegion
    will be [x_len, y_len], where the training region has shape [train_len, x_len, y_len].

    The ArimaModelRegion output is also a FunctionRegion, and it will contain a trained ARIMA model
    in each point P(i, j), trained with the training data at P(i, j) of the training region. The
    ArimaModelRegion can later be applied to a ForecastLengthRegion to obtain the ForecastRegion
    (spatio-temporal region).

    This implementation assumes that the same ARIMA hyper-parameters (p, d, q) are used for all
    the ARIMA models.

    To create an instance of this class by using a training region, use the from_training_region()
    class method. This will produce a different ARIMA model in each point.
    '''
    def __init__(self, train_arima_np, forecast_len):
        '''
        Initializes an instance of this function region, which is made of partial calls to
        train_arima(). Since the output of those calls is an object (an ARIMA model), the dtype
        needs to be set to object.

        The forecast length needs to be known now, because apply_to will create an instance of
        ArimaModelRegion (a FunctionRegionSeries), and functions that output series (the forecast)
        need to know the output length in advance.
        '''
        super(ArimaTrainingRegion, self).__init__(train_arima_np, dtype=object)
        self.forecast_len = forecast_len

    def apply_to(self, spt_region):
        '''
        Decorate the default behavior of FunctionRegionScalar.

        The ouput of the parent behavior is to create a SpatialRegion, but we want to create an
        ArimaModelRegion instance. This method will build on the previous result, which already
        has a trained model in each point.
        '''
        # get result from parent behavior
        spatial_region = super(ArimaTrainingRegion, self).apply_to(spt_region)

        # count and log missing models, iterate to find them
        missing_count = 0
        for (point, arima_model) in self:
            if isinstance(arima_model, FailedArima):
                missing_count += 1
                log_msg = 'Could not train ARIMA {} for point {}'
                self.logger.warn(log_msg.format(str(arima_params), point))

        if missing_count:
            self.logger.warn('Missing ARIMA models: {}' % str(missing_count))
        else:
            self.logger.info('ARIMA was trained in all points successfully.')

        # return ArimaModelRegion instead of SpatialRegion, in order to create forecasts
        # Since ArimaModelRegion is a FunctionRegionSeries, it requires the length of its output
        # series, that is forecast_len.

        # TODO: can this approach support clusters later?!
        return ArimaModelRegion(spatial_region.as_numpy, output_len=self.forecast_len)

    @classmethod
    def from_training_region(cls, training_region, arima_params, forecast_len=FORECAST_LENGTH):
        '''
        Creates an instance of this class. This will produce a different ARIMA model in each point.

        training_region
            spatio-temporal region used as training dataset
        arima_params
            ArimaParams hyper-parameters
        '''

        # output shape is given by training shape
        (_, x_len, y_len) = training_region.shape

        # the function signature to be handled by regions can only receive a series
        # so we need to use partial here
        arima_with_params = partial(train_arima, arima_params)

        # the function is applied over all the region, to train models with the same
        # hyperparameters over the training region
        train_arima_np = arrays_util.copy_value_as_matrix_elements(arima_with_params, x_len, y_len)

        return ArimaTrainingRegion(train_arima_np, forecast_len)


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
            # TODO catch None model here, and get rid of FailArima?
            return model_at_point.forecast(self.output_len)[0]

        return forecast_from_model


def evaluate_forecast_errors_arima(spt_region, arima_params, forecast_len=FORECAST_LENGTH,
                                   centroid=None):
    '''
    Compare the following forecast errors:

    1. Build one ARIMA model for each point, forecast FORECAST_LENGTH days and compute the error
       of each forecast. This will create an error region (error_region_each).
       Combine the errors using distance_measure.combine to obtain a single metric for the
       prediction error. (error_region_each.combined_error)

    2. Choose the model with the smallest local prediction error among the points of the region
       (min_local_arima). Then use that model to forecast the entire region, and obtain the
       combined error. (error_region_min_local.combined_error)

    3. Find the centroid of the region, then use the ARIMA model of *that* point to forecast the
        entire region (errro_region_centroid), and obtain the combined error
        (errro_region_centroid.combined_error).

    If centroid is provided, skip its calculation (it does not depend on ARIMA, only on the
    dataset)
    '''
    logger = log_util.logger_for_me(evaluate_forecast_errors_arima)
    logger.info('Using (p, d, q) = %s' % (arima_params,))
    (training_region, test_region) = train.split_region_in_train_test(spt_region, forecast_len)

    #
    # ARIMA for each point in the region
    #

    # a function region with produces trained models when applied to a training region
    arima_trainings = ArimaTrainingRegion.from_training_region(training_region, arima_params,
                                                               forecast_len)
    arima_models_each = arima_trainings.apply_to(training_region)

    # do a forecast: pass an (empty!) region. This region will control the iteration though.
    empty_region_2d = training_region.empty_region_2d()

    # use the ARIMA models to forecast for their respective points
    forecast_region_each = arima_models_each.apply_to(empty_region_2d)

    #error_region_each = forecast.ErrorRegion.create_from_forecasts(forecast_region_each,
    #                                                              test_region)

    # calculate forecast error using MASE
    error_region_each = ErrorRegionMASE(forecast_region_each, test_region, training_region)
    overall_error_each = error_region_each.overall_error
    logger.info('Combined error from all ARIMAs: {}'.format(overall_error_each))

    #
    # ARIMA with minimum local error
    #

    point_min_error = error_region_each.point_with_min_error()
    logger.info('Point at which ARIMA has min local error: %s' % str(point_min_error))

    # create a forecast region that is made of the forecast series with min local error,
    # repeated all over the region
    forecast_region_min_local = forecast_region_each.repeat_point(point_min_error)

    # error for the entire region using that ARIMA model
    error_region_min_local = ErrorRegionMASE(forecast_region_min_local, test_region,
                                             training_region)
    overall_error_min_local = error_region_min_local.overall_error
    log_msg = 'Error from ARIMA with min local error: {}'
    logger.info(log_msg.format(overall_error_min_local))

    # find the centroid point of the region, use its ARIMA for forecasting
    if centroid:
        logger.info('Using pre-established centroid: %s' % str(centroid))
    else:
        centroid = spt_region.get_centroid(distance_measure=DistanceByDTW())

    # the model at the centroid

    # create a forecast region that is made of the forecast series at the centroid,
    # repeated all over the region
    forecast_region_centroid = forecast_region_each.repeat_point(centroid)

    # error_region_centroid = forecast.ErrorRegion.create_from_forecasts(forecast_region_centroid,
    #                                                                    test_region)
    error_region_centroid = ErrorRegionMASE(forecast_region_centroid, test_region,
                                            training_region)
    overall_error_centroid = error_region_centroid.overall_error
    logger.info('Error from centroid ARIMA: {}'.format(overall_error_centroid))

    overall_errors = (overall_error_each, overall_error_min_local, overall_error_centroid)
    return (centroid, training_region, forecast_region_each, test_region, arima_models_each,
            overall_errors)


def plot_one_arima(training_region, forecast_region, test_region, arima_region):
    point = Point(0, 0)
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
    from spta.region import Region, SpatioTemporalRegion, SpatioTemporalRegionMetadata
    nordeste_small_md = SpatioTemporalRegionMetadata(
        'nordeste_small', Region(43, 50, 85, 95), series_len=365, ppd=1, last=True)
    spt_region = SpatioTemporalRegion.from_metadata(nordeste_small_md)

    # region has known centroid
    centroid = Point(5, 4)

    # use these ARIMA parameters
    arima_params = ArimaParams(1, 1, 1)
    forecast_len = 8

    (centroid, training_region, forecast_region, test_region, arima_region, _) =\
        evaluate_forecast_errors_arima(spt_region, arima_params, forecast_len, centroid)
    plot_one_arima(training_region, forecast_region, test_region, arima_region)
