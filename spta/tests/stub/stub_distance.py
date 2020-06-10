import numpy as np

from spta.distance import DistanceBetweenSeries


def stub_distance_matrix():
    '''
    Returns a small, fictitous distance matrix that matches a 2x3 region (6x6 distance matrix)
    Has the correct properties: triangular matrix and diagonal values are zeros.
    '''
    # manual dataset
    # least sum of distances at index 1 (total distance = 40)
    # second least sum of distanes at index 2 (total distance = 54)
    distances_0 = np.array((0, 11, 12, 13, 14, 15))   # total 65
    distances_1 = np.array((11, 0,  8,  5,  7,  9))   # total 40
    distances_2 = np.array((12, 8,  0,  9, 12, 13))   # total 54
    distances_3 = np.array((13, 5,  9,  0, 16, 18))   # total 61
    distances_4 = np.array((14, 7, 12, 16,  0, 14))   # total 63
    distances_5 = np.array((15, 9, 13, 18, 14, 0))    # total 69

    # shape: x_len = 2, y_len = 3
    distance_matrix = np.empty((6, 6))
    distance_matrix[:, 0] = distances_0
    distance_matrix[:, 1] = distances_1
    distance_matrix[:, 2] = distances_2
    distance_matrix[:, 3] = distances_3
    distance_matrix[:, 4] = distances_4
    distance_matrix[:, 5] = distances_5

    return distance_matrix


def stub_distance_measure():
    '''
    Uses the distance_matrix provided in stub_distance_matrix.
    '''
    distance_measure = DistanceBetweenSeries()
    distance_measure.distance_matrix = stub_distance_matrix()
    return distance_measure
