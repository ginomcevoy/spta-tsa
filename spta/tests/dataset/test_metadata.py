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


class TestAveragePentads(unittest.TestCase):
    '''Unit tests for spta.dataset.samples.AveragePentads.'''

    def setUp(self):
        self.dataset_4spd = stub_region.numpy_3d_4spd_stub()

    def test_convert_same(self):
        # given a time_to_series that represents average pentads
        time_to_series = metadata.AveragePentads()

        # when asked to convert to average pentads
        result = time_to_series.convert(self.dataset_4spd, metadata.AveragePentads())

        # then the dataset is unchanged
        expected = result
        stub_region.verify_result_is_expected(self, result, expected)

    def test_convert_different(self):
        # given a time_to_series that represents average pentads
        time_to_series = metadata.AveragePentads()

        # given converting to something else then error
        with self.assertRaises(AssertionError):
            time_to_series.convert(self.dataset_4spd, metadata.SamplesPerDay(1))

    def test_repr(self):
        # given a time_to_series that represents average pentads
        time_to_series = metadata.AveragePentads()

        # when repr
        result = repr(time_to_series)

        # then
        self.assertEqual(result, 'avg_pentads')


class TestTemporalMetadata(unittest.TestCase):
    '''Unit tests for spta.dataset.samples.TemporalMetadata.'''

    def test_bad_request(self):
        # given a temporal md and an inconsistent request
        temporal_md = metadata.TemporalMetadata(1979, 2015, metadata.SamplesPerDay(4))
        (year_start_request, year_end_request) = (2002, 2000)

        # when doing a bad request, then error
        with self.assertRaises(ValueError):
            temporal_md.years_to_series_interval(year_start_request, year_end_request)

    def test_years_to_series_interval_last_two_years(self):
        # given a temporal md with spd=4
        samples_per_day = 4
        temporal_md = metadata.TemporalMetadata(1979, 2015, metadata.SamplesPerDay(samples_per_day))

        # when doing a request for last two years
        (year_start_request, year_end_request) = (2014, 2015)
        result = temporal_md.years_to_series_interval(year_start_request, year_end_request)

        # then we have +36 days resulting from 9 leap years!
        expected = (51136, 54056)
        self.assertEqual(result, expected)

    def test_repr(self):
        # given a temporal md with spd=4
        samples_per_day = 4
        temporal_md = metadata.TemporalMetadata(1979, 2015, metadata.SamplesPerDay(samples_per_day))

        # when asked for repr
        result = repr(temporal_md)

        # then
        expected = '1979_2015_4spd'
        self.assertEqual(result, expected)

    def test_temporal_metadata_equality(self):
        # given two temporal md with spd=4
        samples_per_day = 4
        temporal_md1 = metadata.TemporalMetadata(1979, 2015, metadata.SamplesPerDay(samples_per_day))
        temporal_md2 = metadata.TemporalMetadata(1979, 2015, metadata.SamplesPerDay(samples_per_day))

        # when compared then are equal
        self.assertEqual(temporal_md1, temporal_md2)
