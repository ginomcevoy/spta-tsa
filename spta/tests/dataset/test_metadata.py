'''Unit tests for spta.dataset.samples module.'''

import numpy as np
import unittest

from spta.tests.stub import stub_region

from spta.dataset import metadata


class TestSamplesPerDay(unittest.TestCase):
    '''Unit tests for spta.dataset.samples.SamplesPerDay.'''

    def setUp(self):
        self.dataset_4spd = stub_region.numpy_3d_4spd_stub()

    def test_convert_same(self):
        # given a time_to_series that represents 4spd
        dataset_spd = 4
        time_to_series = metadata.SamplesPerDay(dataset_spd)

        # when asked to convert to 4spd
        conversion_spd = 4
        result = time_to_series.convert(self.dataset_4spd, metadata.SamplesPerDay(conversion_spd))

        # then the dataset is unchanged
        expected = result
        stub_region.verify_result_is_expected(self, result, expected)

    def test_convert_averaged(self):
        # given a time_to_series that represents 4spd
        dataset_spd = 4
        time_to_series = metadata.SamplesPerDay(dataset_spd)

        # when asked to convert to 1spd
        conversion_spd = 1
        result = time_to_series.convert(self.dataset_4spd, metadata.SamplesPerDay(conversion_spd))

        # then the dataset is averaged (expected built manually)
        expected = np.empty((1, 2, 3))
        expected[:, 0, 0] = np.array((3,))
        expected[:, 0, 1] = np.array((8,))
        expected[:, 0, 2] = np.array((13,))
        expected[:, 1, 0] = np.array((18,))
        expected[:, 1, 1] = np.array((23,))
        expected[:, 1, 2] = np.array((28,))
        stub_region.verify_result_is_expected(self, result, expected)

    def test_repr(self):
        # given a time_to_series that represents 4spd
        dataset_spd = 4
        time_to_series = metadata.SamplesPerDay(dataset_spd)

        # when repr
        result = repr(time_to_series)

        # then
        self.assertEqual(result, '4spd')
