import numpy as np
import time

from statsmodels.tsa.arima_model import ARIMA
from collections import namedtuple


from . import region

TRAINING_FRACTION = 0.7

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

    def __init__(self, training_subset, test_subset, arima_region):
        self.training_subset = training_subset
        self.test_subset = test_subset
        self.arima_region = arima_region

    # spatio + arima on each point

    @classmethod
    def train(cls, spatio_temp_region, arima_params, arima_func=apply_arima):
        '''
        Trains an ARIMA model for each time series in the region.
        Splits the dataset in training and test data

        returns:
            training_region:    SpatioTemporalRegion for training data
            test_region:        SpatioTemporalRegion for training data
            arima_region:    2D region where each point is an ARIMA model that has been fitted with
                training data of that point
        '''
        (training_subset, test_subset) = split_region_in_train_test(spatio_temp_region)

        training_array = training_subset.as_numpy
        (x_len, y_len, training_len) = training_array.shape

        # reshape training set to get an array of x*y, each element a time series
        # NOTE: for some reason this is returning a numpy array of a matrix (Zx1), where Z = x*y
        training_dataset_1d = training_array.reshape(x_len * y_len, training_len)
        training_dataset_by_point = np.split(training_dataset_1d, x_len * y_len)

        print('training_dataset_by_point: %s points' % len(training_dataset_by_point))

        # run arima for each point
        arima_model_by_point = [
            # get rid of the "matrix" inconvenience with [0]
            arima_func(time_series[0], arima_params)
            for time_series
            in training_dataset_by_point
        ]
        print('arima_model_by_point: %s points' % len(arima_model_by_point))

        # print(arima_model_by_point)

        # rebuild the region using the known shape
        arima_2d = np.array(arima_model_by_point).reshape(x_len, y_len)
        arima_region = ArimaSpatialRegion(arima_2d)
        return ArimaForEachPoint(training_subset, test_subset, arima_region)


def split_region_in_train_test(spatio_temp_region):
    series_len = spatio_temp_region.series_len()
    training_size = int(series_len * TRAINING_FRACTION)

    training_interval = region.TimeInterval(0, training_size)
    test_interval = region.TimeInterval(training_size, series_len)

    training_subset = spatio_temp_region.interval_subset(training_interval)
    test_subset = spatio_temp_region.interval_subset(test_interval)

    return (training_subset, test_subset)


if __name__ == '__main__':

    t_start = time.time()

    sptr = region.SpatioTemporalRegion.load_4years()
    small_region = sptr.get_small()

    arima_params = ArimaParams(1, 1, 1)
    arimas = ArimaForEachPoint.train(small_region, arima_params)

    print('train %s:' % (arimas.training_subset.shape,))
    print('test %s:' % (arimas.test_subset.shape,))
    print('arima: %s' % (arimas.arima_region.shape,))

    for pvalues in arimas.arima_region.pvalues_by_point():
        print(pvalues)
