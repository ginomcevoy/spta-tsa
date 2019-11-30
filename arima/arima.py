import numpy as np
import pandas as pd
import time

# from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from collections import namedtuple

from matplotlib import pyplot as plt

from . import region
from . import dataset as ds
from . import validate

# TRAINING_FRACTION = 0.7
TEST_DAYS = 7
TEST_SAMPLES = ds.POINTS_PER_DAY * TEST_DAYS

ArimaParams = namedtuple('ArimaParams', 'p d q')


def apply_arima(time_series, arima_params):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.
    '''
    model = ARIMA(time_series, order=(arima_params.p, arima_params.d, arima_params.q))
    model_fit = model.fit(disp=0)
    return model_fit


class ArimaSpatialRegion(region.SpatioTemporalRegion):

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
        arima_2d = np.array(self.arima_models_1d).reshape(x_len, y_len, 1)
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


def create_forecast_region_one_model(arima_model, region, series_len=TEST_SAMPLES):
    '''
    Create a forecast over a region with *one* ARIMA model. This will just replicate the
    same forecast result over the specified region. Returns a SpatioTemporalRegion, where the
    time series is the forecast with length series_len.
    '''
    # forecast the same number samples that we have for testing by default
    forecast_one = arima_model.forecast(steps=series_len)
    return region.SpatioTemporalRegion.copy_series_over_region(forecast_one, region)


def split_region_in_train_test(spatio_temp_region):
    series_len = spatio_temp_region.series_len()

    # divide series in training and test, training come first in the series
    training_size = series_len - TEST_SAMPLES
    training_interval = region.TimeInterval(0, training_size)
    test_interval = region.TimeInterval(training_size, series_len)

    training_subset = spatio_temp_region.interval_subset(training_interval)
    test_subset = spatio_temp_region.interval_subset(test_interval)

    return (training_subset, test_subset)


def plot_one_arima(training_region, forecast_region, test_region):
    first = region.Point(2, 0)
    train_first = training_region.series_at(first)
    forecast_first = forecast_region.series_at(first)
    test_first = test_region.series_at(first)

    (_, _, training_len) = training_region.shape
    (_, _, test_len) = test_region.shape
    test_index = np.arange(training_len, training_len + test_len)

    train_series = pd.Series(train_first)
    forecast_series = pd.Series(forecast_first, index=test_index)
    test_series = pd.Series(test_first, index=test_index)

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

    sptr = region.SpatioTemporalRegion.load_4years()
    small_region = sptr.get_small()

    # arima_params = ArimaParams(1, 1, 1)
    arima_params = ArimaParams(3, 2, 1)

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

    return (training_region, test_region, forecast_region, arima_region)


if __name__ == '__main__':

    t_start = time.time()

    sptr = region.SpatioTemporalRegion.load_4years()
    small_region = sptr.get_small()

    # arima_params = ArimaParams(1, 1, 1)
    arima_params = ArimaParams(3, 2, 1)

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

    # plot_one_arima(training_region, forecast_region, test_region)

    error_region = validate.ErrorRegion.create_from_forecasts(forecast_region, test_region)
    print(error_region)
    print(error_region.combined_rmse)

