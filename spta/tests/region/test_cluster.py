import numpy as np
import unittest

from spta.region import Point
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster
from spta.region.partition import PartitionRegionCrisp

from spta.tests.stub import stub_region


class TestSpatioTemporalCluster(unittest.TestCase):
    '''
    Unit tests for spta.region.temporal.SpatioTemporalCluster.
    '''

    def setUp(self):
        self.numpy_dataset = stub_region.numpy_3d_stub()

        self.series_0_0 = self.numpy_dataset[:, 0, 0]
        self.series_1_0 = self.numpy_dataset[:, 1, 0]
        self.series_1_2 = self.numpy_dataset[:, 1, 2]

    def test_constructor(self):
        '''
        tests that constructor initializes internal objects (diamond problem)
        '''

        # given
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 0
        partition = PartitionRegionCrisp(mask.reshape((2, 3)), k)

        # when
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, partition, cluster_index, None)

        # then all constructor elements should have been assigned
        self.assertIsNotNone(cluster)
        self.assertTrue(cluster.as_numpy is self.numpy_dataset)
        self.assertTrue(cluster.partition is partition)
        self.assertEquals(cluster.cluster_index, cluster_index)
        self.assertTrue(cluster.logger is not None)

    def test_series_at_point_in_mask(self):

        # given
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 1
        partition = PartitionRegionCrisp(mask.reshape((2, 3)), k)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, partition, cluster_index, None)

        # when asked for points in mask
        result_0_0 = cluster.series_at(Point(0, 0))
        result_1_0 = cluster.series_at(Point(1, 0))
        result_1_2 = cluster.series_at(Point(1, 2))

        # then the series are retrieved
        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(result_0_0, self.series_0_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, self.series_1_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, self.series_1_2))

    def test_series_at_point_not_in_mask(self):

        # given
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 1
        spatial_mask = PartitionRegionCrisp(mask.reshape((2, 3)), k)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, spatial_mask, cluster_index, None)

        # when asked for a point not in mask, should throw error
        with self.assertRaises(ValueError):
            cluster.series_at(Point(0, 1))

        with self.assertRaises(ValueError):
            cluster.series_at(Point(0, 2))

        with self.assertRaises(ValueError):
            cluster.series_at(Point(1, 1))

    def test_iterator(self):

        # given
        mask = np.array([1, 0, 0, 1, 0, 1])
        k = 2
        cluster_index = 1
        spatial_mask = PartitionRegionCrisp(mask.reshape((2, 3)), k)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, spatial_mask, cluster_index, None)

        # when iterating
        iterated_points = []
        iterated_series = []
        for (point, series) in cluster:
            iterated_points.append(point)
            iterated_series.append(series)

        # then only three points should have been iterated
        self.assertEquals(len(iterated_points), 3)
        self.assertEquals(len(iterated_series), 3)

        self.assertEquals(iterated_points[0], Point(0, 0))
        self.assertEquals(iterated_points[1], Point(1, 0))
        self.assertEquals(iterated_points[2], Point(1, 2))

        self.assertIsNone(np.testing.assert_array_equal(iterated_series[0], self.series_0_0))
        self.assertIsNone(np.testing.assert_array_equal(iterated_series[1], self.series_1_0))
        self.assertIsNone(np.testing.assert_array_equal(iterated_series[2], self.series_1_2))
