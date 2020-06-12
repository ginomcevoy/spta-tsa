import numpy as np
import unittest

from spta.region import Point
from spta.region.spatial import SpatialCluster
from spta.region.temporal import SpatioTemporalCluster
from spta.tests.stub import stub_region

'''
Integration test: create a spatio-temporal cluster, then apply a function to it.
'''

class TestFunctionRegionOnSpatioTemporalCluster(unittest.TestCase):

    def test_apply_scalar(self):

        # given a clustering of a sptr
        spt_region = stub_region.spatio_temporal_region_stub()
        members = np.array([2, 0, 1, 2, 0, 2])

        # given the function that calculates the mean of a series in each point
        mean_function_region = stub_region.stub_mean_function_scalar()

        # when getting cluster with cluster index 2 and applying the function to it
        cluster_2 = SpatioTemporalCluster.from_crisp_clustering(spt_region, members,
                                                                cluster_index=2)
        mean_cluster_2 = mean_function_region.apply_to(cluster_2)

        # then result is a SpatialCluster
        self.assertTrue(isinstance(mean_cluster_2, SpatialCluster))

        # then values at the cluster are the means
        result_0_0 = mean_cluster_2.value_at(Point(0, 0))
        result_1_0 = mean_cluster_2.value_at(Point(1, 0))
        result_1_2 = mean_cluster_2.value_at(Point(1, 2))

        mean_0_0 = np.mean(spt_region.series_at(Point(0, 0)))
        mean_1_0 = np.mean(spt_region.series_at(Point(1, 0)))
        mean_1_2 = np.mean(spt_region.series_at(Point(1, 2)))

        self.assertEquals(result_0_0, mean_0_0)
        self.assertEquals(result_1_0, mean_1_0)
        self.assertEquals(result_1_2, mean_1_2)

        # then an iteration should work over the cluster
        sum_of_means = 0
        for (point, a_mean) in mean_cluster_2:
            sum_of_means += a_mean
        self.assertEquals(sum_of_means, mean_0_0 + mean_1_0 + mean_1_2)

        # then a point outside the cluster should be invalid
        with self.assertRaises(ValueError):
            mean_cluster_2.value_at(Point(0, 1))

    def test_apply_series(self):

        # given a clustering of a sptr
        spt_region = stub_region.spatio_temporal_region_stub()
        members = np.array([2, 0, 1, 2, 0, 2])

        # given the function that reverses each series
        reverse_function_region = stub_region.stub_reverse_function_series()

        # when getting cluster with cluster index 2 and applying the function to it
        cluster_2 = SpatioTemporalCluster.from_crisp_clustering(spt_region, members,
                                                                cluster_index=2)
        reversed_cluster = reverse_function_region.apply_to(cluster_2, output_len=3)

        # then result is a SpatialTemporalCluster
        self.assertTrue(isinstance(reversed_cluster, SpatioTemporalCluster))

        # then series at the cluster are the reversed of the original
        result_0_0 = reversed_cluster.series_at(Point(0, 0))
        result_1_0 = reversed_cluster.series_at(Point(1, 0))
        result_1_2 = reversed_cluster.series_at(Point(1, 2))

        reversed_0_0 = spt_region.series_at(Point(0, 0))[::-1]
        reversed_1_0 = spt_region.series_at(Point(1, 0))[::-1]
        reversed_1_2 = spt_region.series_at(Point(1, 2))[::-1]

        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(result_0_0, reversed_0_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, reversed_1_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, reversed_1_2))

        # then a point outside the cluster should be invalid
        with self.assertRaises(ValueError):
            reversed_cluster.series_at(Point(0, 1))
