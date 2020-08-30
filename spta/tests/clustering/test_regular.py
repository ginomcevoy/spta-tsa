'''
Unit tests for spta.clustering.regular module.
'''

import unittest

from spta.clustering.regular import RegularClusteringMetadata


class TestRegularClusteringMetadata(unittest.TestCase):
    '''
    Unit tests for spta.clustering.regular.RegularClusteringMetadata.
    '''

    def test_from_repr_2(self):
        # given
        repr_str = 'regular_k2'

        # when
        instance = RegularClusteringMetadata.from_repr(repr_str)

        # then
        self.assertEqual(instance.__class__.__name__, 'RegularClusteringMetadata')
        self.assertEqual(instance.k, 2)

    def test_from_repr_150(self):
        # given
        repr_str = 'regular_k150'

        # when
        instance = RegularClusteringMetadata.from_repr(repr_str)

        # then
        self.assertEqual(instance.__class__.__name__, 'RegularClusteringMetadata')
        self.assertEqual(instance.k, 150)
