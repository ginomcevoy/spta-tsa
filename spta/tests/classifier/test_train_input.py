'''
Unit tests for spta.classifier.train_input module.
'''

from spta.region import Region, Point
from spta.region.metadata import SpatioTemporalRegionMetadata

from spta.classifier.train_input import MedoidsChoiceMinDistance
from spta.clustering.suite import ClusteringSuite
from spta.distance.dtw import DistanceByDTW

from spta.tests.stub import stub_clustering

import unittest


class TestMedoidsChoiceMinDistance(unittest.TestCase):
    '''
    Unit tests for train_input.MedoidsChoiceMinDistance class.
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

    def test_choose_medoid(self):
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
        instance = MedoidsChoiceMinDistance(region_metadata=self.region_metadata,
                                            distance_measure=self.distance_measure)
        result = instance.choose_medoid(suite_result, point)

        # then
        self.assertEqual(result, ('kmedoids_k3_seed0_lite', 1, Point(48, 89)))

    def test_csv_filepath(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()
        output_home = 'outputs'

        # given input for <count> random points
        count = 10
        random_seed = 0

        instance = MedoidsChoiceMinDistance(region_metadata=self.region_metadata,
                                            distance_measure=self.distance_measure)

        # when
        result = instance.csv_filepath(output_home, kmedoids_suite, count, random_seed)

        # then
        expected = 'outputs/nordeste_small_2015_2015_1spd/dtw/' \
            'random_point_dist_medoid__kmedoids-quick_count10_seed0.csv'
        self.assertEqual(result, expected)
