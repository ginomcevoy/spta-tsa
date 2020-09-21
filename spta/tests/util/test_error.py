import numpy as np
import unittest

from spta.util import error as error_util


class TestMASE(unittest.TestCase):
    '''
    Unit tests for MASE calculation of forecast error.
    '''

    def test_zeros(self):

        # given series with zeros
        forecast_series = np.array([0, 0, 0])
        observation_series = np.array([0, 0, 0])
        training_series = np.array([1, 2, 3])

        # when MASE
        result = error_util.mase(forecast_series, observation_series, training_series)

        # then MASE returns 0
        self.assertEqual(result, 0)

    def test_exact_forecast(self):

        # given exact forecast
        forecast_series = np.array([4, 5, 6])
        observation_series = np.array([4, 5, 6])
        training_series = np.array([1, 2, 3])

        # when MASE
        result = error_util.mase(forecast_series, observation_series, training_series)

        # then MASE returns 0
        self.assertEqual(result, 0)

    def test_forecast_constant_error(self):

        # given a forecast with constant error (et = 0.1 for each point)
        forecast_series = np.array([4.1, 5.1, 5.9])
        observation_series = np.array([4, 5, 6])
        training_series = np.array([1, 2, 3])

        # when MASE
        result = error_util.mase(forecast_series, observation_series, training_series)

        # then MASE returns mean((0.1, 0.1, 0.1)) / (1/2) * 2 = 0.1
        np.testing.assert_almost_equal(result, 0.1)

    def test_forecast_varying_error(self):

        # given a forecast with different errors et = (0.1, 0.2, 0.3)
        forecast_series = np.array([4.1, 5.2, 5.7])
        observation_series = np.array([4, 5, 6])
        training_series = np.array([1, 2, 3])

        # when MASE
        result = error_util.mase(forecast_series, observation_series, training_series)

        # then MASE returns mean((0.1, 0.2, 0.3)) / (1/2) * 2 = 0.2
        np.testing.assert_almost_equal(result, 0.2)
