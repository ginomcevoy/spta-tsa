'''
Unit tests for spta.region.centroid module.
'''

import numpy as np
import unittest

from spta.region import Point
from spta.region.centroid import CalculateCentroid
from spta.region.partition import PartitionRegionCrisp
from spta.region.temporal import SpatioTemporalCluster

from spta.tests.stub import stub_region, stub_distance


class TestCalculateCentroid(unittest.TestCase):
    '''
    Unit tests for centroid.CalculateCentroid class.
    '''

    def setUp(self):
        # use stub distance matrix
        distance_measure = stub_distance.stub_distance_measure()
        self.calculate_centroid = CalculateCentroid(distance_measure)

    def test_find_centroid_and_distances_region(self):

        # given a spatio-temporal region
        spt_region = stub_region.spatio_temporal_region_stub()

        # when
        centroid, _ = self.calculate_centroid.find_centroid_and_distances(spt_region)

        # then index 1 in the matrix has the least distance, so Point(0, 1)
        expected = Point(0, 1)
        self.assertEquals(centroid, expected)

    def test_find_centroid_and_distances_clusters(self):

        # given a spatio-temporal region divided into two clusters
        # cluster0: index 1 has least sum of distances: (1->*16*, 4->28, 5->23)
        # cluster1: index 2 has least sum of distances: (0->25, 2->*21*, 3->22)
        spt_region = stub_region.spatio_temporal_region_stub()
        _, x_len, y_len = spt_region.shape
        members = np.array([1, 0, 1, 1, 0, 0])

        partition = PartitionRegionCrisp.from_membership_array(members, x_len, y_len)
        (cluster0, cluster1) = partition.create_all_spt_clusters(spt_region)

        # when finding centroid of cluster0
        centroid0, _ = self.calculate_centroid.find_centroid_and_distances(cluster0)

        # then index 1 -> Point(0, 1)
        expected0 = Point(0, 1)
        self.assertEquals(centroid0, expected0)

        # when finding centroid of cluster1
        centroid1, _ = self.calculate_centroid.find_centroid_and_distances(cluster1)

        # then index 2 -> Point(0, 2)
        expected1 = Point(0, 2)
        self.assertEquals(centroid1, expected1)
