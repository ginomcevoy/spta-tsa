'''
Unit tests for spta.region.partition module.
'''
import unittest

from spta.region import Point
from spta.region.partition import PartitionRegionCrisp
from spta.tests.stub import stub_partition, stub_region


class TestPartitionRegionCrisp(unittest.TestCase):
    '''
    Unit tests for partition.PartitionRegionCrisp class.
    '''

    def setUp(self):
        self.partition_np = stub_partition.crisp_membership_stub()
        self.k = 3

    def test_membership_of_point_indices_empty(self):

        # given a cluster partition and no points
        partition = PartitionRegionCrisp(self.partition_np, self.k)
        points = []

        # when asking for memberships
        result = partition.membership_of_points(points)

        # then result is empty
        expected = []
        self.assertEqual(result, expected)

    def test_membership_of_points_single(self):

        # given a cluster partition and one point
        partition = PartitionRegionCrisp(self.partition_np, self.k)
        points = [Point(0, 0)]

        # when asking for memberships
        result = partition.membership_of_points(points)

        # then result is membership of point
        expected = [0]
        self.assertEqual(result, expected)

    def test_membership_of_points_three(self):

        # given a cluster partition and three points
        partition = PartitionRegionCrisp(self.partition_np, self.k)
        points = [Point(0, 0), Point(2, 4), Point(1, 1)]

        # when asking for memberships
        result = partition.membership_of_points(points)

        # then result is memberships of points
        expected = [0, 2, 1]
        self.assertEqual(result, expected)

    def test_create_spt_cluster(self):
        # given a cluster partition and an index
        spt_region = stub_region.spt_region_stub_2_4_5()
        partition = PartitionRegionCrisp(self.partition_np, self.k)
        cluster_index = 0

        # when creating a cluster
        cluster0 = partition.create_spt_cluster(spt_region, cluster_index)

        # then the cluster has the given index
        self.assertEqual(cluster0.cluster_index, cluster_index)

        # then the size is correctly calculated
        self.assertEqual(cluster0.cluster_len, 6)

    def test_create_spatial_cluster(self):
        # given a cluster partition and an index
        spatial_region = stub_region.spatial_region_stub()
        partition = PartitionRegionCrisp(self.partition_np, self.k)
        cluster_index = 0

        # when creating a cluster
        cluster0 = partition.create_spatial_cluster(spatial_region, cluster_index)

        # then the cluster has the given index
        self.assertEqual(cluster0.cluster_index, cluster_index)

        # then the size is correctly calculated
        self.assertEqual(cluster0.cluster_len, 6)

    def test_merge_clusters_2d(self):

        # given a cluster partition and the resulting clusters
        spatial_region = stub_region.spatial_region_stub()
        partition = PartitionRegionCrisp(self.partition_np, self.k)

        spatial_clusters = [
            partition.create_spatial_cluster(spatial_region, i)
            for i
            in range(0, self.k)
        ]

        # when merging the clusters
        merged = partition.merge_clusters_2d(spatial_clusters)

        # then merged is a region that has the same values as the input spatial region
        for point, orig_value in spatial_region:
            self.assertEqual(orig_value, merged.value_at(point))

    def test_merge_with_representatives_2d(self):

        # given a cluster partition and the resulting clusters
        spatial_region = stub_region.spatial_region_stub()
        partition = PartitionRegionCrisp(self.partition_np, self.k)

        spatial_clusters = [
            partition.create_spatial_cluster(spatial_region, i)
            for i
            in range(0, self.k)
        ]

        # given the representatives of the three clusters
        # 0 -> (0, 3) = 9; 1 -> (1, 0) = 25; 2 -> (1, 3) = 64
        representatives = [Point(0, 3), Point(1, 0), Point(1, 3)]

        # when merging using values at the representatives
        merged = partition.merge_with_representatives_2d(spatial_clusters, representatives)

        # then the resulting region only has three different values
        # the values correspond to the cluster mask
        self.assertEqual(merged.value_at(Point(0, 0)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(0, 1)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(0, 2)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(0, 3)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(0, 4)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(1, 0)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(1, 1)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(1, 2)), 64)  # cluster2
        self.assertEqual(merged.value_at(Point(1, 3)), 64)  # cluster2
        self.assertEqual(merged.value_at(Point(1, 4)), 64)  # cluster2
        self.assertEqual(merged.value_at(Point(2, 0)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(2, 1)), 64)  # cluster2
        self.assertEqual(merged.value_at(Point(2, 2)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(2, 3)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(2, 4)), 64)  # cluster2
        self.assertEqual(merged.value_at(Point(3, 0)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(3, 1)), 9)  # cluster0
        self.assertEqual(merged.value_at(Point(3, 2)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(3, 3)), 25)  # cluster1
        self.assertEqual(merged.value_at(Point(3, 4)), 64)  # cluster2
