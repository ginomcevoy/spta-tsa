'''
Unit tests for spta.clustering.factory module
'''

import unittest

from spta.clustering.factory import ClusteringMetadataFactory


class TestClusteringMetadataFactory(unittest.TestCase):
    '''
    Unit tests for ClusteringMetadataFactory
    '''

    def setUp(self):
        self.factory = ClusteringMetadataFactory()

    def test_build_regular(self):
        # given
        name = 'regular'
        k = 2

        # when building regular metadata
        metadata = self.factory.instance(name, k)

        # then type, name and k match
        self.assertEqual(metadata.__class__.__name__, 'RegularClusteringMetadata')
        self.assertEqual(metadata.name, 'regular')
        self.assertEqual(metadata.k, 2)

    def test_build_kmedoids(self):
        # given
        name = 'kmedoids'
        k = 2
        random_seed = 1

        # when building kmedoids metadata
        metadata = self.factory.instance(name, k, random_seed=random_seed)

        # then type, name, k and random_seed match
        self.assertEqual(metadata.__class__.__name__, 'KmedoidsClusteringMetadata')
        self.assertEqual(metadata.name, 'kmedoids')
        self.assertEqual(metadata.k, 2)
        self.assertEqual(metadata.random_seed, 1)

    def test_build_kmedoids_with_option(self):
        # given
        name = 'kmedoids'
        k = 2
        random_seed = 1
        mode = 'lite'
        max_iter = 5000

        # when building kmedoids metadata
        metadata = self.factory.instance(name, k, random_seed=random_seed, mode=mode,
                                         max_iter=max_iter)

        # then type, name, k and random_seed match
        self.assertEqual(metadata.__class__.__name__, 'KmedoidsClusteringMetadata')
        self.assertEqual(metadata.name, 'kmedoids')
        self.assertEqual(metadata.k, 2)
        self.assertEqual(metadata.random_seed, 1)
        self.assertEqual(metadata.mode, 'lite')
        self.assertEqual(metadata.max_iter, 5000)

