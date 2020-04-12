import logging
import time
import numpy as np

from spta.region import Point
from spta.region.temporal import SpatioTemporalRegion
from spta.region.error import ErrorRegion

from . import arima

TEST_SAMPLES = 7
CENTROID = Point(5, 15)
ITERATIONS = 1000
RESULT_OUTPUT = 'raw/performance.npy'


def evaluate_arima_performance(arima_params, training_region, test_region, forecast_len, centroid):
    '''
    Time the forecasting of an ARIMA model to establish a performance profile.
    We assume that the centroid is known by known to avoid excessive times.
    '''
    log = logging.getLogger()

    centroid_series = training_region.series_at(centroid)
    centroid_arima = arima.apply_arima(centroid_series, arima_params, centroid)

    if centroid_arima is None:
        log.error('ARIMA with %s on centroid %s was None!' % (arima_params, str(centroid)))
        return np.array([arima_params.p, arima_params.d, arima_params.q, np.nan, np.nan])

    # do a forecast over the region
    t_start = time.time()
    for i in range(0, ITERATIONS):
        centroid_arima.forecast(forecast_len)
    t_stop = time.time()
    elapsed = t_stop - t_start

    # find the forecast error
    forecast_using_centroid = arima.create_forecast_region_one_model(centroid_arima,
                                                                     training_region)
    error_region_centroid = ErrorRegion.create_from_forecasts(forecast_using_centroid, test_region)
    prediction_error = error_region_centroid.combined_error

    msg = 'Forecast took %s for %s iterations, error %s'
    log.info(msg % (elapsed, str(ITERATIONS), str(prediction_error)))

    return np.array([arima_params.p, arima_params.d, arima_params.q, elapsed, prediction_error])


def evaluate_arima_performance_parameters(p_values, d_values, q_values, training_region,
                                          test_region, forecast_len, centroid):
    '''
    Sweep for ARIMA values, gather performance and error
    '''

    total_evaluations = len(p_values) * len(d_values) * len(q_values)
    results = np.empty((total_evaluations, 5))

    index = 0
    for p in p_values:
        for d in d_values:
            for q in q_values:
                arima_params = arima.ArimaParams(p, d, q)
                results[index] = evaluate_arima_performance(arima_params, training_region,
                                                            test_region, forecast_len, centroid)
                index = index + 1

    return results


if __name__ == '__main__':
    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    sptr = SpatioTemporalRegion.load_sao_paulo()
    # region_interest = sptr.get_small()
    region_interest = sptr

    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)

    (training_region, test_region) = arima.split_region_in_train_test(region_interest,
                                                                      TEST_SAMPLES)

    results = evaluate_arima_performance_parameters(p_values, d_values, q_values, training_region,
                                                    test_region, TEST_SAMPLES, CENTROID)

    print(results)
    np.save(RESULT_OUTPUT, results)

    # arima_params = arima.ArimaParams(1, 1, 1)
    # evaluate_arima_performance(arima_params, training_region, test_region,
    #                            TEST_SAMPLES, CENTROID)
