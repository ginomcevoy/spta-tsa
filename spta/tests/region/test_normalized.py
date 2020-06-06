'''
Unit tests for spta.region.SpatioTemporalNormalized
'''

import numpy as np
import unittest

from spta.tests.stub import stub_region
from spta.region import Point
from spta.region.temporal import SpatioTemporalNormalized


class TestSpatioTemporalNormalized(unittest.TestCase):

    def test_normalization(self):
        # given a spatio-temporal region
        sptr = stub_region.spatio_temporal_region_stub()
        series_len, x_len, y_len = sptr.shape

        # when normalizing the region
        norm_region = SpatioTemporalNormalized(sptr)

        # then the series are normalized
        # this stub normalizes all series to the same values...
        result_0_0 = norm_region.series_at(Point(0, 0))
        result_1_0 = norm_region.series_at(Point(1, 0))
        result_1_2 = norm_region.series_at(Point(1, 2))
        norm_series_all = np.array([0, 0.5, 1])

        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(result_0_0, norm_series_all))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, norm_series_all))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, norm_series_all))

        # then normalization_min stores all min values
        expected_mins = np.array([1, 4, 7, 11, 14, 17])
        expected_mins_2d = np.reshape(expected_mins, (x_len, y_len))

        actual_mins = norm_region.normalization_min.numpy_dataset
        self.assertIsNone(np.testing.assert_array_equal(expected_mins_2d, actual_mins))

        # then normalization_max stores all max values
        expected_max = np.array([3, 6, 9, 13, 16, 19])
        expected_max_2d = np.reshape(expected_max, (x_len, y_len))

        actual_max = norm_region.normalization_max.numpy_dataset
        self.assertIsNone(np.testing.assert_array_equal(expected_max_2d, actual_max))
