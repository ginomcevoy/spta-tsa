'''
Unit tests for spta.util.arrays module.
'''

import numpy as np
import unittest

from spta.util import arrays as arrays_util


class TestSquarePartitioning(unittest.TestCase):
    '''
    Unit tests for spta.util.arrays.regular_partitioning function.
    '''

    def test_regular_partitioning_region_too_small(self):
        # given k = 4*3, x_len < 3
        x_len = 2
        y_len = 20
        k = 12

        # then fail because too small
        with self.assertRaises(ValueError):
            arrays_util.regular_partitioning(x_len, y_len, k)

    def test_regular_partitioning_exact(self):
        # when region can be exactly partitioned
        x_len = 20
        y_len = 12
        k = 12

        # when
        matrix = arrays_util.regular_partitioning(x_len, y_len, k)

        # then we get 12 regions each of size (20/4) x (12/3) = (5x4)

        self.assertEquals(matrix.shape, (20, 12))

        # test region i=0 (first)
        matrix_0 = matrix[0:5, 0:4]
        expected_0 = np.repeat(0, 20).reshape(5, 4)

        # this idiom compares numpy arrays
        self.assertIsNone(np.testing.assert_array_equal(matrix_0, expected_0))

        # test region i = 1
        matrix_1 = matrix[0:5, 4:8]
        expected_1 = np.repeat(1, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_1, expected_1))

        # test region i = 5
        matrix_5 = matrix[5:10, 8:12]
        expected_5 = np.repeat(5, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_5, expected_5))

        # test region i=11 (last)
        matrix_11 = matrix[15:20, 8:12]
        expected_11 = np.repeat(11, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_11, expected_11))

    def test_regular_partitioning_y_len_a_bit_larger(self):

        # when region cannot be exactly partitioned because y_len is a bit too large
        x_len = 20
        y_len = 13
        k = 12

        # when
        matrix = arrays_util.regular_partitioning(x_len, y_len, k)

        # then we get 12 regions, but the last column is a bit wider...

        self.assertEquals(matrix.shape, (20, 13))

        # test region i = 1
        matrix_1 = matrix[0:5, 4:8]
        expected_1 = np.repeat(1, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_1, expected_1))

        # test region i = 2, last column of first row
        matrix_2 = matrix[0:5, 8:13]  # 5*5 = 25 points
        expected_2 = np.repeat(2, 25).reshape(5, 5)
        self.assertIsNone(np.testing.assert_array_equal(matrix_2, expected_2))

        # test region i = 5
        matrix_5 = matrix[5:10, 8:12]
        expected_5 = np.repeat(5, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_5, expected_5))

        # test region i=11 (last)
        matrix_11 = matrix[15:20, 8:13]
        expected_11 = np.repeat(11, 25).reshape(5, 5)
        self.assertIsNone(np.testing.assert_array_equal(matrix_11, expected_11))

    def test_regular_partitioning_x_len_a_bit_larger(self):

        # when region cannot be exactly partitioned because x_len is a bit too large
        x_len = 21
        y_len = 12
        k = 12

        # when
        matrix = arrays_util.regular_partitioning(x_len, y_len, k)

        # then we get 12 regions, but the last row is a bit taller...

        self.assertEquals(matrix.shape, (21, 12))

        # test region i = 2, last column of first row, normal case
        matrix_2 = matrix[0:5, 8:12]
        expected_2 = np.repeat(2, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_2, expected_2))

        # test region i = 8, still normal
        matrix_8 = matrix[10:15, 8:12]
        expected_8 = np.repeat(8, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_8, expected_8))

        # test region i = 9, at edge and should be taller
        matrix_9 = matrix[15:21, 0:4]
        expected_9 = np.repeat(9, 24).reshape(6, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_9, expected_9))

        # test region i=11 (last), at edge and should be taller
        matrix_11 = matrix[15:21, 8:12]
        expected_11 = np.repeat(11, 24).reshape(6, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_11, expected_11))

    def test_regular_partitioning_both_a_bit_larger(self):

        # when region cannot be exactly partitioned because x_len and y_len are a bit too large
        x_len = 21
        y_len = 13
        k = 12

        # when
        matrix = arrays_util.regular_partitioning(x_len, y_len, k)

        # then we get 12 regions, but the last row is a bit taller and last col a bit wider

        self.assertEquals(matrix.shape, (21, 13))

        # test region i = 1
        matrix_1 = matrix[0:5, 4:8]
        expected_1 = np.repeat(1, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_1, expected_1))

        # test region i = 2, last column of first row is a bit wider
        matrix_2 = matrix[0:5, 8:13]
        expected_2 = np.repeat(2, 25).reshape(5, 5)
        self.assertIsNone(np.testing.assert_array_equal(matrix_2, expected_2))

        # test region i = 9, last row should be a bit taller
        matrix_9 = matrix[15:21, 0:4]
        expected_9 = np.repeat(9, 24).reshape(6, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_9, expected_9))

        # test region i=11 (last), at edge and should be both wider and taller
        matrix_11 = matrix[15:21, 8:13]
        expected_11 = np.repeat(11, 30).reshape(6, 5)
        self.assertIsNone(np.testing.assert_array_equal(matrix_11, expected_11))

    def test_regular_partitioning_y_len_a_bit_shorter(self):

        # when region cannot be exactly partitioned because y_len is a bit too small
        x_len = 20
        y_len = 11
        k = 12

        # when
        matrix = arrays_util.regular_partitioning(x_len, y_len, k)

        # then we get 12 regions, but the last column is less wide...

        self.assertEquals(matrix.shape, (20, 11))

        # test region i = 1
        matrix_1 = matrix[0:5, 4:8]
        expected_1 = np.repeat(1, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_1, expected_1))

        # test region i = 2, last column of first row is less wide
        matrix_2 = matrix[0:5, 8:11]  # 5*3 = 15 points
        expected_2 = np.repeat(2, 15).reshape(5, 3)
        self.assertIsNone(np.testing.assert_array_equal(matrix_2, expected_2))

        # test region i = 4 normal
        matrix_4 = matrix[5:10, 4:8]
        expected_4 = np.repeat(4, 20).reshape(5, 4)
        self.assertIsNone(np.testing.assert_array_equal(matrix_4, expected_4))

        # test region i=11 (last) should be less wide
        matrix_11 = matrix[15:20, 8:11]
        expected_11 = np.repeat(11, 15).reshape(5, 3)
        self.assertIsNone(np.testing.assert_array_equal(matrix_11, expected_11))
