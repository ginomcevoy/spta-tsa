import numpy as np
import unittest

from spta.region import Point
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster
from spta.region.mask import MaskRegionCrisp

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
        label = 0
        mask_region = MaskRegionCrisp(mask.reshape((2, 3)), label)

        # when
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, mask_region, None)

        # then all constructor elements should have been assigned
        self.assertIsNotNone(cluster)
        self.assertTrue(cluster.as_numpy is self.numpy_dataset)
        self.assertTrue(cluster.mask_region is mask_region)
        self.assertEquals(cluster.label, label)
        self.assertTrue(cluster.logger is not None)

    def test_series_at_point_in_mask(self):

        # given
        mask = np.array([1, 0, 0, 1, 0, 1])
        label = 1
        mask_region = MaskRegionCrisp(mask.reshape((2, 3)), label)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, mask_region, None)

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
        label = 1
        spatial_mask = MaskRegionCrisp(mask.reshape((2, 3)), label)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, spatial_mask, None)

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
        label = 1
        spatial_mask = MaskRegionCrisp(mask.reshape((2, 3)), label)
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        cluster = SpatioTemporalCluster(spt_region, spatial_mask, None)

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

    def test_from_crisp_clustering(self):

        # given
        spt_region = SpatioTemporalRegion(self.numpy_dataset)
        members = np.array([2, 0, 1, 2, 0, 2])
        label = 2

        # when
        cluster = SpatioTemporalCluster.from_crisp_clustering(spt_region, members, label, None)

        # then a proper cluster is obtained. with mask where members = 2
        self.assertEquals(cluster.label, 2)

        # when asked for points in mask
        result_0_0 = cluster.series_at(Point(0, 0))
        result_1_0 = cluster.series_at(Point(1, 0))
        result_1_2 = cluster.series_at(Point(1, 2))

        # then the series are retrieved
        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(result_0_0, self.series_0_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_0, self.series_1_0))
        self.assertIsNone(np.testing.assert_array_equal(result_1_2, self.series_1_2))

        # when asked for a point not in mask, should throw error
        with self.assertRaises(ValueError):
            cluster.series_at(Point(0, 1))

        with self.assertRaises(ValueError):
            cluster.series_at(Point(0, 2))

        with self.assertRaises(ValueError):
            cluster.series_at(Point(1, 1))
