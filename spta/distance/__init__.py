'''
Handle distances between series, in particular using Dynamic Time Warping (DTW).
'''

import logging
import numpy as np

from spta.util import log as log_util


class DistanceBetweenSeries(log_util.LoggerMixin):

    def __init__(self):
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

        log_msg = 'Loaded distance matrix for region {} using: {}'
        self.logger.info(log_msg.format(str(expected_region), filename))
        return self.distance_matrix

    def load_distance_matrix_md(self, sptr_metadata):
        '''
        Given metadata for a spatio-temporal region, loads its DTW pre-computed distance matrix.

        TODO: the sptr_metadata.distances_filename should indicate that it was computed with DTW!
        '''
        return self.load_distance_matrix_2d(sptr_metadata.distances_filename,
                                            sptr_metadata.region)

    def distances_to_point(self, spt_region, point, all_point_indices, use_distance_matrix=True):
        '''
        Given a spatio-temporal region and a point in the region, compute the distances between
        each point in the region and the specified point.

        Need to know all points in the region, to subset the distance_matrix appropriately.
        This is a non-issue for normal regions, but needs to be done using all_point_indices to
        support cluster iteration.

        all_point_indices is required here so that it will not be called within this code.
        Attempting to call spt_region.all_point_indices now will fail (nested loop with same
        iterator!) if this method is within an external loop. Example: when calculating the
        centroid by looping over points.

        If use_distance_matrix = True, will try to load a pre-computed distance matrix if possible.
        If matrix is not available, it will compute it using subclass-specific behavior.
        Using use_distance_matrix requires region metadata.

        TODO: support use_distance_matrix = False!
        NOTE: Don't reimplement this in subclasses!
        '''
        if not use_distance_matrix:
            raise NotImplementedError('use_distance_matrix=False not supported yet!')

        if self.distance_matrix is None:

            try:
                # can we load a saved distance matrix?
                # this requires the metadata of the region
                self.load_distance_matrix_md(spt_region.region_metadata)

            except Exception as err:
                log_msg = 'Calculating distances because saved distances not available: {}'
                self.logger.warn(log_msg.format(err))

                # if distance matrix is not available, go ahead and calculate it
                self.compute_distance_matrix(spt_region)

        # use the distance matrix to calculate distances to the point
        return self._distances_to_point_with_matrix(spt_region, point, all_point_indices)

    def _distances_to_point_with_matrix(self, spt_region, point, all_point_indices):
        '''
        Given a spatio-temporal region and a point in the region, compute the distances between
        each point in the region and the specified point.

        Uses pre-computed distance matrix to obtain the distances.

        all_point_indices is required here so that it will not be called within this code.
        Attempting to call spt_region.all_point_indices now will fail (nested loop with same
        iterator!) if this method is within an external loop. Example: when calculating the
        centroid by looping over points.
        '''
        assert self.distance_matrix is not None

        _, _, y_len = spt_region.shape

        # subset the matrix using the point index and the indices of the points in the region
        point_index = point.x * y_len + point.y
        distances_to_point = self.distance_matrix[point_index, all_point_indices]

        return distances_to_point
