'''
Unit tests for spta.model.knn module.
'''
import unittest
import numpy as np

from spta.model import knn
from spta.distance.rmse import DistanceByRMSE


class TestFindKnearestNeighbors(unittest.TestCase):
    '''
    Unit tests for knn.find_k_nearest_neighbors function
    '''

    def setUp(self):
        self.distance_measure = DistanceByRMSE()

    def test_fail_if_a_neighbor_has_wrong_length(self):

        # given an array and two neighbors but one has wrong size
        array = np.array([1, 2, 3, 4])
        k = 2
        possible_neighbors = (
            np.array([1, 2, 3, 5]),
            np.array([1, 5, 3, 4, 0])
        )

        # then calling function results in error
        with self.assertRaises(ValueError):
            knn.find_k_nearest_neighbors(array, k, possible_neighbors, self.distance_measure)

    def test_find_k2_when_two_are_available(self):

        # given an array, k=2 and only two possible neighbors
        array = np.array([1, 2, 3, 4])
        k = 2
        possible_neighbors = (
            np.array([1, 2, 3, 5]),
            np.array([1, 5, 3, 4])
        )

        # when
        result = knn.find_k_nearest_neighbors(array, k, possible_neighbors, self.distance_measure)

        # then the indices of the two neighbors are returned
        expected = (0, 1)
        self.assertEqual(result, expected)

    def test_find_k1_example(self):

        # given an array, k=1 and two possible neighbors
        array = np.array([1, 2, 3, 4])
        k = 1
        possible_neighbors = (
            np.array([1, 2, 3, 5]),
            np.array([1, 2, 3, 4])
        )

        # when
        result = knn.find_k_nearest_neighbors(array, k, possible_neighbors, self.distance_measure)

        # then the neighbor which is identical is returned
        expected = (1,)
        self.assertEqual(result, expected)

    def test_find_k3_example(self):

        # given an array, k=3 and 8 possible neighbors
        array = np.array([1, 2, 3, 4, 5])
        k = 3
        possible_neighbors = (
            np.array([1.9, 2.1, 3.0, 3.4, 5.1]),
            np.array([1.0, 2.1, 3.0, 4.0, 5.1]),  # OK
            np.array([1.0, 2.1, 3.5, 4.0, 5.3]),
            np.array([1.0, 2.0, 1.0, 4.0, 5.0]),
            np.array([1.0, 2.0, 2.9, 4.1, 4.9]),  # OK
            np.array([0.9, 2.1, 3.1, 3.9, 5.1]),  # OK
            np.array([1.0, 2.1, 3.0, 5.0, 5.1]),
            np.array([0.0, 2.1, 3.0, 5.0, 5.1]),
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        )

        # when
        result = knn.find_k_nearest_neighbors(array, k, possible_neighbors, self.distance_measure)

        # then the 3 neighbors are found, in undefined order
        expected = (1, 4, 5)
        self.assertEqual(sorted(result), sorted(expected))


class TestPredictFutureValuesWithKNN(unittest.TestCase):
    '''
    Unit tests for knn.predict_future_values_with_knn function
    '''

    def test_predict_future_values_with_knn(self):

        # given a monotone series and some parameters
        max_val = 20
        time_series = np.array(range(0, max_val))
        k = 3
        forecast_len = 5
        distance_measure = DistanceByRMSE()

        # when
        result = knn.predict_future_values_with_knn(time_series, k, forecast_len, distance_measure)

        # then the three nearest neighbors of the last window are the previous three windows
        n1 = np.array(range(max_val - forecast_len - 3, max_val - 3))
        n2 = np.array(range(max_val - forecast_len - 2, max_val - 2))
        n3 = np.array(range(max_val - forecast_len - 1, max_val - 1))
        expected = np.mean(np.array([n1, n2, n3]), axis=0)
        self.assertIsNone(np.testing.assert_array_equal(result, expected))
