'''
Unit tests for spta.clustering.suite module.
'''

import unittest

from spta.region import Region, Point
from spta.region.metadata import SpatioTemporalRegionMetadata
from spta.distance.dtw import DistanceByDTW
from spta.clustering.suite import ClusteringSuite

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
