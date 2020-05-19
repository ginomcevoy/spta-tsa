import numpy as np


def mase(forecast_series, observation_series, training_series):
    '''
    Calculates MASE:

    qt = et / [(1 / n-1) * sum(| Y_i - Y_{i-1}|, i=2, i=n)
    MASE = mean(|qt|)
    '''
    et = forecast_series - observation_series

    n = len(training_series)
    sum_Yi = 0
    for i in range(1, n):
        sum_Yi += np.abs(training_series[i] - training_series[i - 1])
    qt = et / (sum_Yi / (n - 1))
    return np.mean(np.abs(qt))
