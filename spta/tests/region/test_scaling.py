'''
Unit tests for spta.region.scaling.SpatioTemporalScaled
'''

import numpy as np
import unittest

from spta.region import Point
from spta.region.scaling import ScaleFunction
from spta.region.temporal import SpatioTemporalCluster
from spta.region.partition import PartitionRegionCrisp

from spta.tests.stub import stub_region


class TestSpatioTemporalScaled(unittest.TestCase):

    def test_scaling(self):
        # given a spatio-temporal region
        sptr = stub_region.spatio_temporal_region_stub()
        series_len, x_len, y_len = sptr.shape

        # when scaling the region
        scale_function = ScaleFunction(x_len, y_len)
        scaled_region = scale_function.apply_to(sptr, sptr.series_len)

        # then the series are scaled to [0, 1s]
        # this stub scales all series to the same values...
        result_0_0 = scaled_region.series_at(Point(0, 0))
        result_1_0 = scaled_region.series_at(Point(1, 0))
        result_1_2 = scaled_region.series_at(Point(1, 2))
        scaled_series_all = np.array([0, 0.5, 1])

        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(result_0_0, scaled_series_all))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, scaled_series_all))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, scaled_series_all))

        # then scale_min stores all min values
        expected_mins = np.array([1, 4, 7, 11, 14, 17])
        expected_mins_2d = np.reshape(expected_mins, (x_len, y_len))

        actual_mins = scaled_region.scale_min.numpy_dataset
        self.assertIsNone(np.testing.assert_array_equal(expected_mins_2d, actual_mins))

        # then scale_max stores all max values
        expected_max = np.array([3, 6, 9, 13, 16, 19])
        expected_max_2d = np.reshape(expected_max, (x_len, y_len))

        actual_max = scaled_region.scale_max.numpy_dataset
        self.assertIsNone(np.testing.assert_array_equal(expected_max_2d, actual_max))

    def test_scaling_and_descaling(self):
        # given a spatio-temporal region
        sptr = stub_region.spatio_temporal_region_stub()
        series_len, x_len, y_len = sptr.shape

        # when scaling and descaling the region
        scale_function = ScaleFunction(x_len, y_len)
        scaled_region = scale_function.apply_to(sptr, sptr.series_len)
        descaled_region = scaled_region.descale()

        # then we retrieve the original dataset
        original_region = stub_region.spatio_temporal_region_stub()

        result_0_0 = descaled_region.series_at(Point(0, 0))
        result_1_0 = descaled_region.series_at(Point(1, 0))
        result_1_2 = descaled_region.series_at(Point(1, 2))

        expected_0_0 = original_region.series_at(Point(0, 0))
        expected_1_0 = original_region.series_at(Point(1, 0))
        expected_1_2 = original_region.series_at(Point(1, 2))

        self.assertIsNone(np.testing.assert_array_equal(result_0_0, expected_0_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, expected_1_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, expected_1_2))


class TestClusteringAndScaling(unittest.TestCase):

    def test_scaling_then_clustering_then_can_be_descaled(self):

        # given a scaled region
        spt_region = stub_region.spatio_temporal_region_stub()
        series_len, x_len, y_len = spt_region.shape
        scale_function = ScaleFunction(x_len, y_len)
        scaled_region = scale_function.apply_to(spt_region, spt_region.series_len)

        # given a cluster of the scaled region
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 1
        partition = PartitionRegionCrisp(mask.reshape((2, 3)), k)
        cluster = SpatioTemporalCluster(scaled_region, partition, cluster_index, None)

        # then this cluster can be descaled
        self.assertTrue(cluster.has_scaling())

    def test_scaling_then_clustering_then_descaling(self):

        # given a scaled region
        spt_region = stub_region.spatio_temporal_region_stub()
        series_len, x_len, y_len = spt_region.shape
        scale_function = ScaleFunction(x_len, y_len)
        scaled_region = scale_function.apply_to(spt_region, spt_region.series_len)

        # given a cluster of the scaled region
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 1
        partition = PartitionRegionCrisp(mask.reshape((2, 3)), k)
        cluster = SpatioTemporalCluster(scaled_region, partition, cluster_index, None)

        # when descaling the cluster
        descaled_cluster = cluster.descale()

        # then the result is a cluster with the descaled data of the original region
        original_region = stub_region.spatio_temporal_region_stub()

        result_0_0 = descaled_cluster.series_at(Point(0, 0))
        result_1_0 = descaled_cluster.series_at(Point(1, 0))
        result_1_2 = descaled_cluster.series_at(Point(1, 2))

        expected_0_0 = original_region.series_at(Point(0, 0))
        expected_1_0 = original_region.series_at(Point(1, 0))
        expected_1_2 = original_region.series_at(Point(1, 2))

        self.assertIsNone(np.testing.assert_array_equal(result_0_0, expected_0_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, expected_1_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, expected_1_2))

        # then this is a cluster, points outside the cluster are still invalid
        with self.assertRaises(ValueError):
            descaled_cluster.series_at(Point(0, 1))

        with self.assertRaises(ValueError):
            descaled_cluster.series_at(Point(0, 2))

        with self.assertRaises(ValueError):
            descaled_cluster.series_at(Point(1, 1))
