import logging
import numpy as np
import pandas as pd
import time

# from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from collections import namedtuple

from matplotlib import pyplot as plt

from . import region
from . import validate

# TRAINING_FRACTION = 0.7
# TEST_SAMPLES = 28
TEST_SAMPLES = 15


ArimaParams = namedtuple('ArimaParams', 'p d q')


def apply_arima(time_series, arima_params):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.
    '''
    log = logging.getLogger()
    log.debug('Training ARIMA with: %s' % str(arima_params))
    model = ARIMA(time_series, order=(arima_params.p, arima_params.d, arima_params.q))
    model_fit = model.fit(disp=0)
    return model_fit


class ArimaSpatialRegion(region.SpatialRegion):

    def pvalues_by_point(self):
        return [
            model.pvalues
            for model
            in self.as_numpy.flatten()
        ]


class ArimaForEachPoint:

    def __init__(self, training_region, arima_models_1d):
        self.training_region = training_region

        # we store as 1d because we need to operate on it
        self.arima_models_1d = arima_models_1d

    def create_spatial_region(self):
        # rebuild the region using the known shape
        (x_len, y_len, training_len) = self.training_region.shape
        arima_2d = np.array(self.arima_models_1d).reshape(x_len, y_len)
        return ArimaSpatialRegion(arima_2d)

    def create_forecast_region(self, series_len=TEST_SAMPLES):
        '''
        Creates a forecast over a region using the corresponding ARIMA model for each point.
        Returns a SpatioTemporalRegion, where each time series is a different forecast with length
        series_len.
        '''
        # forecast for each point
        forecast_1d = [
            arima_model.forecast(series_len)[0]
            for arima_model
            in self.arima_models_1d
        ]
        print(forecast_1d)

        # rebuild the original region shape, create a spatio temporal object
        (x_len, y_len, _) = self.training_region.shape
        forecast_region_numpy = np.array(forecast_1d).reshape(x_len, y_len, series_len)
        return region.SpatioTemporalRegion(forecast_region_numpy)

    @classmethod
    def train(cls, training_region, arima_params, arima_func=apply_arima):

        training_2d = training_region.as_list

        # run arima for each point
        arima_models_1d = [
            arima_func(time_series, arima_params)
            for time_series
            in training_2d  # training_dataset_by_point
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
    return region.SpatioTemporalRegion.copy_series_over_region(forecast_one, region_3d)


def split_region_in_train_test(spatio_temp_region):
    series_len = spatio_temp_region.series_len()

    # divide series in training and test, training come first in the series
    training_size = series_len - TEST_SAMPLES
    training_interval = region.TimeInterval(0, training_size)
    test_interval = region.TimeInterval(training_size, series_len)

    training_subset = spatio_temp_region.interval_subset(training_interval)
    test_subset = spatio_temp_region.interval_subset(test_interval)

    return (training_subset, test_subset)


def plot_one_arima(training_region, forecast_region, test_region, arima_region):
    point = region.Point(0, 0)
    train_point = training_region.series_at(point)
    forecast_point = forecast_region.series_at(point)
    test_point = test_region.series_at(point)

    (_, _, training_len) = training_region.shape
    (_, _, test_len) = test_region.shape
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

    sptr = region.SpatioTemporalRegion.load_sao_paulo()
    small_region = sptr.get_small()
    log.info('small: %s ' % str(small_region.shape))

    arima_params = ArimaParams(1, 2, 1)
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


def evaluate_forecast_errors_arima(spatio_temp_region, arima_params):
    '''
    Compare the following forecast errors:

    1. Build one ARIMA model for each point, then choose the model with the smallest prediction
        error among the points of the region. Then use that model to forecast the entire region,
        and obtain the combined error.

    2. Find the centroid of the region, then use the ARIMA model of *that* point to forecast the
        entire region, and obtain the combined error.
    '''
    log = logging.getLogger()
    (training_region, test_region) = split_region_in_train_test(spatio_temp_region)

    # train one ARIMA model for each point, get mxn models
    arimasEachPoint = ArimaForEachPoint.train(training_region, arima_params)

    # get a spatial region with an ARIMA model on each point
    arima_region = arimasEachPoint.create_spatial_region()

    # use the ARIMA models to forecast for their respective points
    forecast_region_each = arimasEachPoint.create_forecast_region()

    error_region_each = validate.ErrorRegion.create_from_forecasts(forecast_region_each,
                                                                   test_region)
    log.info('Combined error from all ARIMAs: %s' % error_region_each.combined_rmse)

    point_min_error = error_region_each.point_with_min_error()
    log.info('Point with best ARIMA: %s' % str(point_min_error))

    # use the ARIMA with min error to forecast the entire region
    min_error_arima = arima_region.value_at(point_min_error)
    forecast_using_best = create_forecast_region_one_model(min_error_arima, spatio_temp_region)
    log.debug('forecast_using_best')
    log.debug(forecast_using_best)

    error_region_best = validate.ErrorRegion.create_from_forecasts(forecast_using_best,
                                                                   test_region)
    log.info('Error from best ARIMA: %s' % error_region_best.combined_rmse)

    # find the centroid point of the region, use its ARIMA for forecasting
    centroid = spatio_temp_region.centroid
    # centroid = region.Point(1, 3)
    centroid_arima = arima_region.value_at(centroid)
    log.debug('centroid_arima: %s' % centroid_arima)

    forecast_using_centroid = create_forecast_region_one_model(centroid_arima, spatio_temp_region)
    error_region_centroid = validate.ErrorRegion.create_from_forecasts(forecast_using_centroid,
                                                                       test_region)
    log.info('Error from centroid ARIMA: %s' % error_region_centroid.combined_rmse)


if __name__ == '__main__':

    t_start = time.time()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    # sptr = region.SpatioTemporalRegion.load_4years()
    # example()
    sptr = region.SpatioTemporalRegion.load_sao_paulo()
    small_region = sptr.get_small()

    arima_params = ArimaParams(3, 2, 1)

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
    # print(error_region.combined_rmse)

    evaluate_forecast_errors_arima(small_region, arima_params)
