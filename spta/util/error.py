import numpy as np

from . import arrays


def mase(forecast_series, observation_series, training_series):
    '''
    Calculates MASE:

    qt = et / [(1 / n-1) * sum(| Y_i - Y_{i-1}|, i=2, i=n)
    MASE = mean(|qt|)

    NOTE: This implementation uses the entire training series provided (n = len(training_series))
    '''
    et = forecast_series - observation_series

    n = len(training_series)
    sum_Yi = 0
    for i in range(1, n):
        sum_Yi += np.abs(training_series[i] - training_series[i - 1])
    qt = et / (sum_Yi / (n - 1))
    return np.mean(np.abs(qt))


def smape(forecast_series, observation_series, *args):
    '''
    Calculates sMAPE:

    pt = 200 * (abs(et) / forecast_series + observation_series)
    sMAPE = mean(pt)

    *args is there to support other error functions with more arguments, e.g. mase()
    '''
    et = forecast_series - observation_series
    pt = 200 * (np.abs(et) / (forecast_series + observation_series))
    return np.mean(pt)


def mse(forecast_series, observation_series, *args):
    '''
    Calculates MSE:

    mse = sum((et)^2) / n
    '''
    et = forecast_series - observation_series
    return arrays.mean_squared(et)
