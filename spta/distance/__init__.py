'''
Handle distances between series, in particular using Dynamic Time Warping (DTW).
'''

import logging
import numpy as np


class DistanceBetweenSeries:

    def __init__(self):
        self.logger = logging.getLogger()
        self.distance_matrix = None

    def measure(self, first_series, second_series):
        '''
        Distance between two series. Can be used to evaluate the error between forecast and test.
        '''
        raise NotImplementedError

    def combine(self, distances_for_point):
        '''
        Given several distances related to a single point a region, combine all the distances
        into one meaningful value.
        '''
        raise NotImplementedError

    def compute_distance_matrix(self, temporal_data):
        '''
        Given a spatio-temporal region, calculates and stores the distance matrix, i.e. the
        distances between each two points.

        Works with temporal data: an array of series, or a spatio temporal region.

        The output is a 2d numpy array, with dimensions (x_len*y_len, x_len*y_len). The value
        at (i, j) is the distance between series_i and series_j.
        '''
        raise NotImplementedError

    def load_distance_matrix_2d(self, filename, expected_region):
        '''
        Loads a pre-computed distance matrix from a file for a 2d region.
        The distance matrix is expected to be a 2d matrix [x_len * y_len, x_len * y_len].
        '''

        # read from file
        distance_matrix = np.load(filename)

        # check dimensions
        i_len, j_len = distance_matrix.shape

        expected_x = (expected_region.x2 - expected_region.x1)
        expected_y = (expected_region.y2 - expected_region.y1)
        expected_total_points = expected_x * expected_y

        if i_len != expected_total_points or j_len != expected_total_points:
            err_msg = 'Unexpected distances: expected ({}, {}), got ({}, {})'
            raise ValueError(err_msg.format(expected_total_points, expected_total_points,
                                            i_len, j_len))

        # all good
        self.distance_matrix = distance_matrix
        return self.distance_matrix
