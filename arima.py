import numpy as np
from statsmodels.tsa.arima_model import ARIMA

import dataset as ds
import region


TRAINING_FRACTION = 0.7


def split_series_in_train_test(time_series):
    series_len = time_series.shape[0]
    training_size = int(series_len * TRAINING_FRACTION)

    training_data = time_series[0:training_size]
    test_data = time_series[training_data:series_len]
    return (training_data, test_data)


def split_region_in_train_test(t_region_dataset):
    series_len = t_region_dataset.shape[2]
    training_size = int(series_len * TRAINING_FRACTION)

    training_dataset = t_region_dataset[:, :, 0:training_size]
    test_dataset = t_region_dataset[:, :, training_size:series_len]
    return (training_dataset, test_dataset)


def means(time_series, p, d, q):
    '''
    Just a test
    '''
    return np.mean(time_series)


def apply_arima(time_series, p, d, q):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.
    '''
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit(disp=0)
    return model_fit


def train_arimas_over_region(t_region_dataset, arima_func, p, d, q):
    '''
    Trains an ARIMA model for each time series in the region.
    Splits the dataset in training and test data

    returns:
        training_region: 2D region where each point is a time series of training data
        test_region:     2D region where each point is a time series of test data
        arima_region:    2D region where each point is an ARIMA model that has been fitted with
            training data of that point
    '''
    (x_len, y_len, series_len) = t_region_dataset.shape
    print(t_region_dataset.shape)

    (training_dataset, test_dataset) = split_region_in_train_test(t_region_dataset)

    print('training dataset: %s' % (training_dataset.shape,))
    training_len = training_dataset.shape[2]

    # unpack the training set to get an array of x*y, each element a time series
    # NOTE: for some reason this is returning a numpy array of a matrix (Zx1), where Z = x*y
    training_dataset_1d = training_dataset.reshape(x_len * y_len, training_len)
    training_dataset_by_point = np.split(training_dataset_1d, x_len * y_len)

    print('training_dataset_by_point: %s points' % len(training_dataset_by_point))
    # print(training_dataset_by_point)

    # run arima for each point
    arima_model_by_point = [
        arima_func(time_series[0], p, d, q)  # get rid of the "matrix" inconvenience with [0]
        for time_series
        in training_dataset_by_point
    ]
    print('arima_model_by_point: %s points' % len(arima_model_by_point))

    # print(arima_model_by_point)

    # rebuild the region using the known shape
    arima_region = np.array(arima_model_by_point).reshape(x_len, y_len)
    return (training_dataset, test_dataset, arima_region)


if __name__ == '__main__':

    # test train_arimas_ver_region with squares
    # dummy_region = region.get_dummy()
    # dummy_region = region.get_10p_3x3()

    # load dataset 4years
    dataset = ds.load_with_len(5840)

    # get sao paulo in (x, y, time_series) format
    t_region_dataset = region.transpose_region(region.get_sao_paulo_data(dataset))

    # train ARIMA model on each point of the region
    (training_dataset, test_dataset, arima_region) = train_arimas_over_region(
        t_region_dataset, apply_arima, 1, 1, 1)

    print('train %s:' % (training_dataset.shape,))
    print('test %s:' % (test_dataset.shape,))
    print('arima: %s' % (arima_region.shape,))
