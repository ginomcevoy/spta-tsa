import logging
import numpy as np
import pandas as pd
import time

# from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot as plt

from spta.region import forecast, spatial, temporal
from spta.region import Point, TimeInterval
from spta.util import log as log_util

from . import ArimaParams

# TRAINING_FRACTION = 0.7
# TEST_SAMPLES = 28
TEST_SAMPLES = 8


def apply_arima(time_series, arima_params, point):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.

    If the evaluation fails, return None instead of the model fit.
    '''
    log = logging.getLogger()
    log.debug('Training ARIMA with: %s' % str(arima_params))

    try:
        model = ARIMA(time_series, order=(arima_params.p, arima_params.d, arima_params.q))
        model_fit = model.fit(disp=0)
    except ValueError:
        log.warn('could not train ARIMA %s for point %s' % (str(arima_params), str(point)))
        model_fit = None

    return model_fit


class ArimaSpatialRegion(spatial.SpatialRegion):

    def pvalues_by_point(self):
        return [
            model.pvalues
            for model
            in self.as_numpy.flatten()
        ]


class ArimaForEachPoint(log_util.LoggerMixin):

    def __init__(self, training_region, arima_models_1d):
        self.training_region = training_region

        # we store as 1d because we need to operate on it
        self.arima_models_1d = arima_models_1d

        # count missing models
        missing_count = sum([model is None for model in arima_models_1d])
        self.logger.warn('Missing ARIMA models: %s' % str(missing_count))

    def create_spatial_region(self):
        # rebuild the region using the known shape
        (training_len, x_len, y_len) = self.training_region.shape
        arima_2d = np.array(self.arima_models_1d).reshape(x_len, y_len)
        return ArimaSpatialRegion(arima_2d)

    def create_forecast_region(self, forecast_len=TEST_SAMPLES):
        '''
        Creates a forecast over a region using the corresponding ARIMA model for each point.
        Returns a SpatioTemporalRegion, where each time series is a different forecast with length
        forecast_len.
        '''
        # forecast for each point, missing models get NaN forecast
        # this is a list of (x_len * y_len) arrays, each array has size TEST_SAMPLES
        forecast_arrays = [
            arima_model.forecast(forecast_len)[0]
            if arima_model is not None
            else np.repeat(np.nan, repeats=TEST_SAMPLES)
            for arima_model
            in self.arima_models_1d
        ]
        self.logger.debug('First forecast: {}'.format(forecast_arrays[0]))
        self.logger.debug('Second forecast: {}'.format(forecast_arrays[1]))

        # recover the original region shape
        (_, x_len, y_len) = self.training_region.shape

        # concatenate the list of arrays into a single array,
        # this however will return the wrong order of dimensions
        forecast_wrong_order = np.concatenate(forecast_arrays, axis=0)
        forecast_wrong_shape = (x_len, y_len, forecast_len)
        forecast_region_wrong_order = forecast_wrong_order.reshape(forecast_wrong_shape)

        # to get our forecast right, rearrange the axes
        # shape will now be (forecast_len, x_len, y_len)
        forecast_region_np = np.moveaxis(forecast_region_wrong_order, 2, 0)

        self.logger.debug('Forecast for first point: {}'.format(forecast_region_np[:, 0, 0]))
        self.logger.debug('Forecast for second point: {}'.format(forecast_region_np[:, 0, 1]))

        return temporal.SpatioTemporalRegion(forecast_region_np)

    @classmethod
    def train(cls, training_region, arima_params, arima_func=apply_arima):

        training_2d = training_region.as_list

        # run arima for each point
        arima_models_1d = [
            arima_func(time_series, arima_params, point_index)
            for point_index, time_series
            in enumerate(training_2d)  # training_dataset_by_point
        ]

        return ArimaForEachPoint(training_region, arima_models_1d)

    # @property
    # def region_shape(self):
    #     # return 2d shape
    #     return self.training_region.shape[0:1]


def create_forecast_region_one_model(arima_model, region_3d, series_len=TEST_SAMPLES):
    '''
    Create a forecast over a region with *one* ARIMA model. This will just replicate the
    same forecast result over the specified region (ignoring the 3d component of the region).
    Returns a SpatioTemporalRegion, where the time series is the forecast with length series_len.
    '''
    # forecast the same number samples that we have for testing by default
    forecast_one = arima_model.forecast(steps=series_len)[0]
    return temporal.SpatioTemporalRegion.copy_series_over_region(forecast_one, region_3d)


def split_region_in_train_test(spatio_temp_region, test_len=TEST_SAMPLES):
    series_len = spatio_temp_region.series_len()

    # divide series in training and test, training come first in the series
    training_size = series_len - test_len
    training_interval = TimeInterval(0, training_size)
    test_interval = TimeInterval(training_size, series_len)

    training_subset = spatio_temp_region.interval_subset(training_interval)
    test_subset = spatio_temp_region.interval_subset(test_interval)

    logger = log_util.logger_for_me(split_region_in_train_test)
    logger.debug('training_subset: {}'.format(training_subset.shape))
    logger.debug('test_subset: {}'.format(test_subset.shape))

    return (training_subset, test_subset)


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


def example():
    log = logging.getLogger()

    sptr = temporal.SpatioTemporalRegion.load_sao_paulo()
    small_region = sptr.get_small()
    log.info('small: %s ' % str(small_region.shape))

    # arima_params = ArimaParams(1, 1, 1)
    # arima_params = ArimaParams(1, 2, 1)
    # arima_params = ArimaParams(3, 2, 1)
    # arima_params = ArimaParams(7, 1, 7)

    (training_region, test_region) = split_region_in_train_test(small_region)
    arimasEachPoint = ArimaForEachPoint.train(training_region, arima_params)

    print('train %s:' % (arimasEachPoint.training_region.shape,))
    print('arimas1d: %s' % arimasEachPoint.arima_models_1d)

    # mse = mean_squared_error(A, B)

    # forecast for each point
    forecast_region = arimasEachPoint.create_forecast_region()
    print(forecast_region)
    print(forecast_region.shape)

    arima_region = arimasEachPoint.create_spatial_region()
    for pvalues in arima_region.pvalues_by_point():
        print(pvalues)

    plot_one_arima(training_region, forecast_region, test_region, arima_region)

    return (training_region, test_region, forecast_region, arima_region)


def evaluate_forecast_errors_arima(spatio_temp_region, arima_params, centroid=None):
    '''
    Compare the following forecast errors:

    1. Build one ARIMA model for each point, forecast TEST_SAMPLES days and compute the error
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
    log = logging.getLogger()
    log.info('Using (p, d, q) = %s' % (arima_params,))
    (training_region, test_region) = split_region_in_train_test(spatio_temp_region)

    # train one ARIMA model for each point, get mxn models
    arimasEachPoint = ArimaForEachPoint.train(training_region, arima_params)

    # get a spatial region with an ARIMA model on each point
    arima_region = arimasEachPoint.create_spatial_region()

    # use the ARIMA models to forecast for their respective points
    forecast_region_each = arimasEachPoint.create_forecast_region()

    error_region_each = forecast.ErrorRegion.create_from_forecasts(forecast_region_each,
                                                                   test_region)
    log.info('Combined error from all ARIMAs: %s' % error_region_each.combined_error)

    point_min_error = error_region_each.point_with_min_error()
    log.info('Point with best ARIMA: %s' % str(point_min_error))

    # use the ARIMA with min error to forecast the entire region
    min_local_arima = arima_region.value_at(point_min_error)
    forecast_using_best = create_forecast_region_one_model(min_local_arima, spatio_temp_region)
    # log.debug('forecast_using_best')
    # log.debug(forecast_using_best)

    error_region_min_local = forecast.ErrorRegion.create_from_forecasts(forecast_using_best,
                                                                        test_region)
    log.info('Error from best ARIMA: %s' % error_region_min_local.combined_error)

    # find the centroid point of the region, use its ARIMA for forecasting
    if centroid:
        log.info('Using pre-established centroid: %s' % str(centroid))
    else:
        centroid = spatio_temp_region.centroid

    centroid_arima = arima_region.value_at(centroid)
    # log.debug('centroid_arima: %s' % centroid_arima)

    forecast_using_centroid = create_forecast_region_one_model(centroid_arima, spatio_temp_region)
    error_region_centroid = forecast.ErrorRegion.create_from_forecasts(forecast_using_centroid,
                                                                       test_region)
    log.info('Error from centroid ARIMA: %s' % error_region_centroid.combined_error)

    log.debug('training_region: {}'.format(training_region.shape))
    log.debug('forecast_region: {}'.format(forecast_region_each.shape))
    log.debug('test_region: {}'.format(test_region.shape))
    log.debug('arima_region: {}'.format(arima_region.shape))

    return (centroid, centroid_arima, training_region, forecast_region_each, test_region,
            arima_region)


if __name__ == '__main__':

    t_start = time.time()

    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # sptr = region.SpatioTemporalRegion.load_4years()
    # example()
    region_interest = temporal.SpatioTemporalRegion.load_sao_paulo()
    # region_interest = sptr.get_small()

    # arima_params = ArimaParams(3, 2, 1)
    # arima_params = ArimaParams(7, 1, 7)
    # arima_params = ArimaParams(3, 2, 1)
    # arima_params = ArimaParams(1, 1, 1)
    # arima_params = ArimaParams(2, 0, 2)
    # arima_params = ArimaParams(6, 1, 0)

    # arima_params = ArimaParams(3, 2, 1)

    # (training_region, test_region) = split_region_in_train_test(small_region)
    # arimasEachPoint = ArimaForEachPoint.train(training_region, arima_params)

    # print('train %s:' % (arimasEachPoint.training_region.shape,))
    # print('arimas1d: %s' % arimasEachPoint.arima_models_1d)

    # # mse = mean_squared_error(A, B)

    # # forecast for each point
    # forecast_region = arimasEachPoint.create_forecast_region()
    # print(forecast_region)
    # print(forecast_region.shape)

    # arima_region = arimasEachPoint.create_spatial_region()
    # for pvalues in arima_region.pvalues_by_point():
    #     print(pvalues)

    # plot_one_arima(training_region, forecast_region, test_region)

    # error_region = validate.ErrorRegion.create_from_forecasts(forecast_region, test_region)
    # print(error_region)
    # print(error_region.combined_error)

    # pre-calculated
    centroid = Point(5, 15)
    # evaluate_forecast_errors_arima(region_interest, arima_params, centroid=centroid)

    # Best ARIMA models (Pareto optimal, minimize both time and error):

    # (p, d, q, time, prediction_error)

    # [ 1.          0.          0.          0.27679181 19.3221551 ]
    # [ 2.          0.          0.          0.27396417 19.38779183]
    # [ 2.          0.          2.          0.28502035 18.79402005]
    # [ 4.          0.          0.          0.28283215 19.00455224]
    # [ 6.          1.          0.          0.36856842 18.78800349]

    pdqs = ((1, 0, 0), (2, 0, 0), (2, 0, 2), (4, 0, 0), (6, 1, 0))
    for pdq in pdqs:
        arima_params = ArimaParams(*pdq)
        evaluate_forecast_errors_arima(region_interest, arima_params, centroid=centroid)
