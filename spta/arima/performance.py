import time
import numpy as np

from spta.region import Point
from spta.region.temporal import SpatioTemporalRegion
from spta.region.forecast import ErrorRegion
from spta.region.train import split_region_in_train_test

from spta.util import log as log_util

from . import arima, ArimaParams

TEST_SAMPLES = 7
ITERATIONS = 1000
RESULT_OUTPUT = 'raw/performance.npy'


# TODO make this work with updated ARIMA implementation!


def evaluate_arima_performance(arima_params, training_region, test_region, forecast_len, centroid):
    '''
    Time the forecasting of an ARIMA model to establish a performance profile.
    We assume that the centroid is known by known to avoid excessive times.
    '''
    logger = log_util.logger_for_me(evaluate_arima_performance)

    centroid_series = training_region.series_at(centroid)
    centroid_arima = arima.train_arima(arima_params, centroid_series)

    if centroid_arima is None:
        logger.error('ARIMA with %s on centroid %s was None!' % (arima_params, str(centroid)))
        return np.array([arima_params.p, arima_params.d, arima_params.q, np.nan, np.nan])

    # do a forecast over the region
    t_start = time.time()
    for i in range(0, ITERATIONS):
        forecast_series = centroid_arima.forecast(forecast_len)[0]
    t_stop = time.time()
    elapsed = t_stop - t_start

    # the forecast over the region is just the forecast of the centroid ARIMA, repeated
    # all over the region
    (_, x_len, y_len) = training_region.shape
    forecast_using_centroid = SpatioTemporalRegion.repeat_series_over_region(forecast_series,
                                                                             (x_len, y_len))

    # find the forecast error by using test_region
    error_region_centroid = ErrorRegion.create_from_forecasts(forecast_using_centroid, test_region)
    prediction_error = error_region_centroid.combined_error

    msg = 'Forecast took %s for %s iterations, error %s'
    logger.info(msg % (elapsed, str(ITERATIONS), str(prediction_error)))

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
                arima_params = ArimaParams(p, d, q)
                results[index] = evaluate_arima_performance(arima_params, training_region,
                                                            test_region, forecast_len, centroid)
                index = index + 1

    return results


if __name__ == '__main__':
    log_util.setup_log('DEBUG')

    # get region from metadata
    from spta.region import Region, SpatioTemporalRegionMetadata
    nordeste_small_md = SpatioTemporalRegionMetadata(
        'nordeste_small', Region(43, 50, 85, 95), series_len=365, ppd=1, last=True)
    spt_region = SpatioTemporalRegion.from_metadata(nordeste_small_md)

    # region has known centroid
    centroid = Point(5, 4)

    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)

    (training_region, test_region) = split_region_in_train_test(spt_region, TEST_SAMPLES)

    results = evaluate_arima_performance_parameters(p_values, d_values, q_values, training_region,
                                                    test_region, TEST_SAMPLES, centroid)

    print(results)
    np.save(RESULT_OUTPUT, results)

    # arima_params = arima.ArimaParams(1, 1, 1)
    # evaluate_arima_performance(arima_params, training_region, test_region,
    #                            TEST_SAMPLES, CENTROID)
