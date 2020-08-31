'''
Unit tests for spta.clustering.min_distance module.
'''

from spta.region import Region, Point
from spta.region.metadata import SpatioTemporalRegionMetadata

from spta.clustering.min_distance import FindClusterWithMinimumDistance
from spta.clustering.suite import ClusteringSuite
from spta.distance.dtw import DistanceByDTW

import unittest


class TestFindClusterWithMinimumDistance(unittest.TestCase):
    '''
    Unit tests for min_distance.FindClusterWithMinimumDistance class.
    '''

    def setUp(self):
        # necessary setup to find the CSV file with path
        # resources/nordeste_small_2015_2015_1spd/dtw/clustering__kmedoids-quick.csv
        self.output_home = 'spta/tests/resources'
        self.region_metadata = SpatioTemporalRegionMetadata('nordeste_small',
                                                            Region(43, 50, 85, 95), 2015, 2015, 1,
                                                            scaled=False)
        self.distance_measure = DistanceByDTW()
        self.clustering_suite = ClusteringSuite('quick', 'kmedoids', k=range(2, 3),
                                                random_seed=range(0, 2))

    def test_retrieve_suite_result(self):
        '''
        Expected CSV contents:

        clustering, total_cost, medoids
        kmedoids_k2_seed0_lite 380.062 |(45,86) (47,91) |
        kmedoids_k2_seed1_lite 380.062 |(45,86) (47,91) |
        kmedoids_k3_seed0_lite 351.926 |(45,86) (48,89) (45,92) |
        kmedoids_k3_seed1_lite 351.926 |(45,86) (45,92) (48,89) |
        '''

        # given
        instance = FindClusterWithMinimumDistance(region_metadata=self.region_metadata,
                                                  distance_measure=self.distance_measure,
                                                  clustering_suite=self.clustering_suite)

        # when
        suite_result = instance.retrieve_suite_result_csv(self.output_home)

        # then the results are retrieved
        self.assertTrue('kmedoids_k2_seed0_lite' in suite_result)
        self.assertTrue('kmedoids_k2_seed1_lite' in suite_result)
        self.assertTrue('kmedoids_k3_seed0_lite' in suite_result)
        self.assertTrue('kmedoids_k3_seed1_lite' in suite_result)

        # then the medoids are listed for each clustering metadata, in the expected order
        self.assertEqual(suite_result['kmedoids_k2_seed0_lite'][0], Point(45, 86))
        self.assertEqual(suite_result['kmedoids_k2_seed0_lite'][1], Point(47, 91))

        self.assertEqual(suite_result['kmedoids_k2_seed1_lite'][0], Point(45, 86))
        self.assertEqual(suite_result['kmedoids_k2_seed1_lite'][1], Point(47, 91))

        self.assertEqual(suite_result['kmedoids_k3_seed0_lite'][0], Point(45, 86))
        self.assertEqual(suite_result['kmedoids_k3_seed0_lite'][1], Point(48, 89))
        self.assertEqual(suite_result['kmedoids_k3_seed0_lite'][2], Point(45, 92))

        self.assertEqual(suite_result['kmedoids_k3_seed1_lite'][0], Point(45, 86))
        self.assertEqual(suite_result['kmedoids_k3_seed1_lite'][1], Point(45, 92))
        self.assertEqual(suite_result['kmedoids_k3_seed1_lite'][2], Point(48, 89))

    def test_find_medoid_with_minimum_distance_to_point(self):
        # NOTE this requires a pre-calculated distance matrix!

        # given
        suite_result = {
            'kmedoids_k2_seed0_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k2_seed1_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k3_seed0_lite': [Point(45, 86), Point(48, 89), Point(45, 92)],
            'kmedoids_k3_seed1_lite': [Point(45, 86), Point(45, 92), Point(48, 89)]
        }
        point = Point(5, 5)

        # when
        instance = FindClusterWithMinimumDistance(region_metadata=self.region_metadata,
                                                  distance_measure=self.distance_measure,
                                                  clustering_suite=self.clustering_suite)
        result = instance.find_medoid_with_minimum_distance_to_point(point, suite_result,
                                                                     with_matrix=True)

        # then
        self.assertEqual(result, ('kmedoids_k3_seed0_lite', 1, Point(48, 89)))
