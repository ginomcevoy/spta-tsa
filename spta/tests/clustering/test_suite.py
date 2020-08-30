'''
Unit tests for spta.clustering.suite module.
'''

import unittest

from spta.region import Region
from spta.region.metadata import SpatioTemporalRegionMetadata
from spta.distance.dtw import DistanceByDTW
from spta.clustering.suite import ClusteringSuite


class ClusteringSuiteTest(unittest.TestCase):
    '''
    Unit tests for spta.clustering.suite.ClusteringSuite class.
    '''

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

    def test_csv_filename(self):

        # given a range of k and random_seed for kmedoids metadata
        identifier = 'quick'
        metadata_name = 'kmedoids'
        ks = range(2, 4)  # k=2, k=3
        random_seeds = range(0, 2)  # 0, 1
        kmedoids_suite = ClusteringSuite(identifier, metadata_name, k=ks, random_seed=random_seeds)

        # when
        result = kmedoids_suite.csv_filename()

        # then
        expected = 'clustering__kmedoids-quick.csv'
        self.assertEqual(result, expected)

    def test_csv_filepath(self):

        # given a range of k and random_seed for kmedoids metadata
        identifier = 'quick'
        metadata_name = 'kmedoids'
        ks = range(2, 4)  # k=2, k=3
        random_seeds = range(0, 2)  # 0, 1
        kmedoids_suite = ClusteringSuite(identifier, metadata_name, k=ks, random_seed=random_seeds)

        # given a region metadata and a distance measure
        region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(40, 50, 50, 60),
                                                       2015, 2015, 1, scaled=False)
        distance_measure = DistanceByDTW()
        output_home = 'outputs'

        # when
        result = kmedoids_suite.csv_filepath(output_home, region_metadata, distance_measure)

        # then
        expected = 'outputs/nordeste_small_2015_2015_1spd/dtw/clustering__kmedoids-quick.csv'
        self.assertEqual(result, expected)
