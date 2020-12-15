'''
Unit tests for spta.clustering.suite module.
'''

import unittest

from spta.region import Region, Point
from spta.region.metadata import SpatioTemporalRegionMetadata
from spta.distance.dtw import DistanceByDTW
from spta.clustering.suite import ClusteringSuite, OrganizeClusteringSuite, FindSuiteElbow
from spta.clustering.kmedoids import KmedoidsClusteringMetadata

from spta.tests.stub import stub_clustering


class ClusteringSuiteTest(unittest.TestCase):
    '''
    Unit tests for spta.clustering.suite.ClusteringSuite class.
    '''

    def setUp(self):
        self.output_home = 'spta/tests/resources'
        self.region_metadata = \
            SpatioTemporalRegionMetadata('nordeste_small', Region(40, 50, 50, 60),
                                         2015, 2015, 1, scaled=False)
        self.distance_measure = DistanceByDTW()

    def test_regular_list_2(self):
        # given a range of k for regular metadata
        identifier = 'quick'
        metadata_name = 'regular'
        ks = range(2, 4)  # k=2, k=3
        regular_suite = ClusteringSuite(identifier, metadata_name, k=ks)

        # when
        regular_metadata_list = [metadata for metadata in regular_suite]

        # then the identifier is saved
        self.assertEqual(regular_suite.identifier, identifier)

        # then the list includes the range of k
        ks_in_list = [metadata.k for metadata in regular_metadata_list]
        expected = [2, 3]
        self.assertEqual(sorted(ks_in_list), expected)

        # then the list is of regular instances
        names_in_list = [metadata.name for metadata in regular_metadata_list]
        expected = ['regular', 'regular']
        self.assertEqual(names_in_list, expected)

    def test_regular_list_3(self):
        # given a range of k for regular metadata
        identifier = 'quick2'
        metadata_name = 'regular'
        ks = range(2, 5)  # k=2, k=3, k=4
        regular_suite = ClusteringSuite(identifier, metadata_name, k=ks)

        # when
        regular_metadata_list = [metadata for metadata in regular_suite]

        # then the identifier is saved
        self.assertEqual(regular_suite.identifier, identifier)

        # then the list includes the range of k
        ks_in_list = [metadata.k for metadata in regular_metadata_list]
        expected = [2, 3, 4]
        self.assertEqual(sorted(ks_in_list), expected)

        # then the list is of regular instances
        names_in_list = [metadata.name for metadata in regular_metadata_list]
        expected = ['regular', 'regular', 'regular']
        self.assertEqual(names_in_list, expected)

    def test_kmedoids_list_vary_k(self):
        # given a range of k for kmedoids metadata
        identifier = 'quick'
        metadata_name = 'kmedoids'
        ks = range(2, 4)  # k=2, k=3
        random_seed = 5
        kmedoids_suite = ClusteringSuite(identifier, metadata_name, k=ks, random_seed=random_seed)

        # when
        kmedoids_metadata_list = [metadata for metadata in kmedoids_suite]

        # then the identifier is saved
        self.assertEqual(kmedoids_suite.identifier, identifier)

        # then the list includes the range of k
        ks_in_list = [metadata.k for metadata in kmedoids_metadata_list]
        expected = [2, 3]
        self.assertEqual(sorted(ks_in_list), expected)

        # then the list is of kmedoids instances
        names_in_list = [metadata.name for metadata in kmedoids_metadata_list]
        expected = ['kmedoids', 'kmedoids']
        self.assertEqual(names_in_list, expected)

        # then the all the instances have the same random seed
        seeds_in_list = [metadata.random_seed for metadata in kmedoids_metadata_list]
        expected = [5, 5]
        self.assertEqual(seeds_in_list, expected)

    def test_kmedoids_list_vary_k_vary_seed(self):

        # given a range of k and random_seed for kmedoids metadata
        identifier = 'quick'
        metadata_name = 'kmedoids'
        ks = range(2, 4)  # k=2, k=3
        random_seeds = range(0, 2)  # 0, 1
        kmedoids_suite = ClusteringSuite(identifier, metadata_name, k=ks, random_seed=random_seeds)

        # when
        kmedoids_metadata_list = [metadata for metadata in kmedoids_suite]

        # then the identifier is saved
        self.assertEqual(kmedoids_suite.identifier, identifier)

        # then we have the following:
        # kmedoids_k2_seed0_lite
        # kmedoids_k2_seed1_lite
        # kmedoids_k3_seed0_lite
        # kmedoids_k3_seed1_lite

        metadata_2_0 = kmedoids_metadata_list[0]
        self.assertEqual(metadata_2_0.k, 2)
        self.assertEqual(metadata_2_0.random_seed, 0)

        metadata_2_1 = kmedoids_metadata_list[1]
        self.assertEqual(metadata_2_1.k, 2)
        self.assertEqual(metadata_2_1.random_seed, 1)

        metadata_3_0 = kmedoids_metadata_list[2]
        self.assertEqual(metadata_3_0.k, 3)
        self.assertEqual(metadata_3_0.random_seed, 0)

        metadata_3_1 = kmedoids_metadata_list[3]
        self.assertEqual(metadata_3_1.k, 3)
        self.assertEqual(metadata_3_1.random_seed, 1)

    def test_analysis_csv_filename(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()

        # when
        result = kmedoids_suite.analysis_csv_filename()

        # then
        expected = 'clustering__kmedoids-quick.csv'
        self.assertEqual(result, expected)

    def test_analysis_csv_filepath(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()
        output_home = 'outputs'

        # when
        result = kmedoids_suite.analysis_csv_filepath(output_home, self.region_metadata,
                                                      self.distance_measure)

        # then
        expected = 'outputs/nordeste_small_2015_2015_1spd/dtw/clustering__kmedoids-quick.csv'
        self.assertEqual(result, expected)

    def test_medoid_series_csv_filepath(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()

        # given a region metadata and a distance measure
        region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(40, 50, 50, 60),
                                                       2015, 2015, 1, scaled=False)
        distance_measure = DistanceByDTW()
        output_home = 'outputs'

        # when
        result = kmedoids_suite.medoid_series_csv_filepath(output_home, region_metadata,
                                                           distance_measure)

        # then
        expected = 'outputs/nordeste_small_2015_2015_1spd/dtw/' \
            'medoid_data__kmedoids-quick.csv'
        self.assertEqual(result, expected)

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
        clustering_suite = ClusteringSuite('quick', 'kmedoids', k=range(2, 3),
                                           random_seed=range(0, 2))

        # when
        suite_result = clustering_suite.retrieve_suite_result_csv(output_home=self.output_home,
                                                                  region_metadata=self.region_metadata,
                                                                  distance_measure=self.distance_measure)
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


class TestOrganizeClusteringSuite(unittest.TestCase):
    '''
    Unit tests for spta.clustering.suite.OrganizeClusteringSuite
    '''

    def setUp(self):
        self.organizer = OrganizeClusteringSuite()

    def test_simple(self):
        # given a kmedoids suite with one metadata
        identifier = 'quick'
        metadata_name = 'kmedoids'
        parameter_combinations = {
            'k': (2,),
            'random_seed': (0,),
            'mode': 'lite'
        }
        kmedoids_clustering_suite = ClusteringSuite(identifier, metadata_name, **parameter_combinations)

        # when organizing it
        result = self.organizer.organize_kmedoids_suite(kmedoids_clustering_suite)

        # then there is only this one metadata
        self.assertTrue(0 in result)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0].k, 2)

    def test_three_same_seed(self):
        # given a kmedoids suite with three metadatas same seed
        identifier = 'quick'
        metadata_name = 'kmedoids'
        parameter_combinations = {
            'k': (3, 5, 2),
            'random_seed': (1,),
            'mode': 'lite'
        }
        kmedoids_clustering_suite = ClusteringSuite(identifier, metadata_name, **parameter_combinations)

        # when organizing it
        result = self.organizer.organize_kmedoids_suite(kmedoids_clustering_suite)

        # then all metadatas are in the same seed
        self.assertTrue(1 in result)
        self.assertEqual(len(result[1]), 3)

        # then metadatas are in order
        self.assertEqual(result[1][0].k, 2)
        self.assertEqual(result[1][1].k, 3)
        self.assertEqual(result[1][2].k, 5)

    def test_eight_in_two_seeds(self):

        # given a kmedoids suite with 2 seeds and 4 k values for each seed
        identifier = 'quick'
        metadata_name = 'kmedoids'
        parameter_combinations = {
            'k': (3, 5, 2, 4),
            'random_seed': (1, 2),
            'mode': 'lite'
        }
        kmedoids_clustering_suite = ClusteringSuite(identifier, metadata_name, **parameter_combinations)

        # when organizing it
        result = self.organizer.organize_kmedoids_suite(kmedoids_clustering_suite)

        # then metadatas are split in the two seeds
        self.assertTrue(1 in result)
        self.assertEqual(len(result[1]), 4)
        self.assertTrue(2 in result)
        self.assertEqual(len(result[2]), 4)

        # then metadatas are in order
        self.assertEqual(result[1][0].k, 2)
        self.assertEqual(result[1][1].k, 3)
        self.assertEqual(result[1][2].k, 4)
        self.assertEqual(result[1][3].k, 5)
        self.assertEqual(result[2][0].k, 2)
        self.assertEqual(result[2][1].k, 3)
        self.assertEqual(result[2][2].k, 4)
        self.assertEqual(result[2][3].k, 5)


class TestFindSuiteElbow(unittest.TestCase):
    '''
    Unit tests for spta.clustering.suite.FindSuiteElbow class.
    '''

    def setUp(self):
        # stub version, only usable to find second derivative of inputs
        self.finds_elbow = FindSuiteElbow(None, None, None)

    def test_elbow_only_three_points_elbow_is_middle(self):

        # given only three points
        costs_by_metadata = {
            KmedoidsClusteringMetadata(2): 7.0,
            KmedoidsClusteringMetadata(3): 5.0,
            KmedoidsClusteringMetadata(4): 4.0,
        }
        ordered_metadata_instances = [
            KmedoidsClusteringMetadata(2),
            KmedoidsClusteringMetadata(3),
            KmedoidsClusteringMetadata(4)
        ]

        # when
        result = self.finds_elbow.find_cost_elbow_given_order(costs_by_metadata, ordered_metadata_instances)

        # then elbow metadata is the middle one
        self.assertEqual(result.k, 3)

    def test_elbow_cubic_function_elbow_is_second_to_last(self):

        # given five points following k^3 (d2(k^3)/dk2 = 6k)
        costs_by_metadata = {
            KmedoidsClusteringMetadata(2): 8.0,
            KmedoidsClusteringMetadata(3): 27.0,
            KmedoidsClusteringMetadata(4): 64.0,
            KmedoidsClusteringMetadata(5): 125.0,
            KmedoidsClusteringMetadata(6): 216.0,
        }
        ordered_metadata_instances = [
            KmedoidsClusteringMetadata(2),
            KmedoidsClusteringMetadata(3),
            KmedoidsClusteringMetadata(4),
            KmedoidsClusteringMetadata(5),
            KmedoidsClusteringMetadata(6)
        ]

        # when
        result = self.finds_elbow.find_cost_elbow_given_order(costs_by_metadata, ordered_metadata_instances)

        # then elbow metadata is k = 5 (d2(x^3)/dx2 = 6x, so last value is greatest)
        self.assertEqual(result.k, 5)

    def test_elbow_quadratic_function_elbow_evenly_spaced(self):

        # given five points following f(k) = -2k^4 + 48k^3 + 156k^2 - 5000k + 20000
        # d(f(k))/dk =  -8k^3 + 144k^2 + 312k - 5000
        # d^2(f(k))/dk^2 = -24 * (k^2 - 12k - 13) = -24 * (k-13)(k+1), max for k = 6

        costs_by_metadata = {
            KmedoidsClusteringMetadata(2): 10976.0,  # d2 = 792
            KmedoidsClusteringMetadata(3): 7538.0,  # d2 = 960
            KmedoidsClusteringMetadata(4): 5056.0,  # d2 = 1080
            KmedoidsClusteringMetadata(5): 3650.0,  # d2 = 1152
            KmedoidsClusteringMetadata(6): 3392.0,  # d2 = 1176
            KmedoidsClusteringMetadata(7): 4306.0,  # d2 = 1152
            KmedoidsClusteringMetadata(8): 6368.0,  # d2 = 1080
            KmedoidsClusteringMetadata(9): 9506.0,  # d2 = 960
            KmedoidsClusteringMetadata(10): 13600.0,  # d2 = 792
            KmedoidsClusteringMetadata(11): 18482.0,
            KmedoidsClusteringMetadata(12): 23936.0,
        }
        ordered_metadata_instances = [
            KmedoidsClusteringMetadata(k)
            for k in range(2, 13)
        ]

        # when
        result = self.finds_elbow.find_cost_elbow_given_order(costs_by_metadata, ordered_metadata_instances)

        # then elbow metadata is k = 6 per equations above
        self.assertEqual(result.k, 6)

    def test_elbow_quadratic_function_elbow_unevenly_spaced(self):

        # same as above but we don't have all points

        costs_by_metadata = {
            KmedoidsClusteringMetadata(2): 10976.0,  # d2 = 792
            # KmedoidsClusteringMetadata(3): 7538.0,  # d2 = 960
            KmedoidsClusteringMetadata(4): 5056.0,  # d2 = 1080
            # KmedoidsClusteringMetadata(5): 3650.0,  # d2 = 1152
            KmedoidsClusteringMetadata(6): 3392.0,  # d2 = 1176
            KmedoidsClusteringMetadata(7): 4306.0,  # d2 = 1152
            KmedoidsClusteringMetadata(8): 6368.0,  # d2 = 1080
            KmedoidsClusteringMetadata(9): 9506.0,  # d2 = 960
            # KmedoidsClusteringMetadata(10): 13600.0,  # d2 = 792
            # KmedoidsClusteringMetadata(11): 18482.0,
            KmedoidsClusteringMetadata(12): 23936.0,
        }
        ordered_metadata_instances = [
            KmedoidsClusteringMetadata(k)
            for k in (2, 4, 6, 7, 8, 9, 12)
        ]

        # when
        result = self.finds_elbow.find_cost_elbow_given_order(costs_by_metadata, ordered_metadata_instances)

        # then elbow metadata is k = 6 per equations above
        self.assertEqual(result.k, 6)
