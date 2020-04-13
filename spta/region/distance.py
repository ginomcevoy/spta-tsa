import logging
import numpy as np
from tslearn.metrics import dtw

from spta.region import Point
from spta.util import arrays as arrays_util


class DistanceBetweenSeries:

    def __init__(self):
        self.logger = logging.getLogger()

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


class DistanceByDTW(DistanceBetweenSeries):
    '''
    Use Dynamic Time Warping
    '''

    def measure(self, first_series, second_series):
        if np.isnan(first_series).any() or np.isnan(second_series).any():
            return np.nan
        else:
            return dtw(first_series, second_series)

    def combine(self, distances_for_point):
        '''
        Given many distances, combine them to provide a single metric for the distance between
        one series and a list of series.
        For DTW, we use the Root Mean Squared.
        '''
        return arrays_util.root_mean_squared(distances_for_point)

    def compute_distance_matrix(self, temporal_data):
        '''
        Given a spatio-temporal region, calculates and stores the distance matrix, i.e. the
        distances between each two points.

        Works with temporal data: an array of series, or a spatio temporal region.

        The output is a 2d numpy array, with dimensions (x_len*y_len, x_len*y_len). The value
        at (i, j) is the distance between series_i and series_j.
        '''

        if temporal_data.ndim == 2:
            # assume array of series
            distance_matrix = self.compute_distance_matrix_series_array(temporal_data)

        elif temporal_data.ndim == 3:
            # assume SpatioTemporalRegion instance
            distance_matrix = self.compute_distance_matrix_sptr(temporal_data)

        else:
            err_msg = 'Cannot work with supplied temporal_data: {}'
            raise ValueError(err_msg.format(type(temporal_data)))

        # save it in this instance for reusability
        self.distance_matrix = distance_matrix
        return distance_matrix

    def compute_distance_matrix_series_array(self, X):
        series_n, _ = X.shape
        distance_matrix = np.empty((series_n, series_n))

        # iterate the series
        for i in range(0, series_n):

            series_i = X[i, :]

            # calculate the distances to all other series
            self.logger.debug('Calculating distances at: {}...'.format(i))
            distances_for_i = [
                self.measure(series_i, other_series)
                for other_series
                in X
            ]
            self.logger.debug('Got: {}'.format(str(distances_for_i)))
            distance_matrix[i, :] = distances_for_i

        return distance_matrix

    def compute_distance_matrix_sptr(self, spatio_temporal_region):

        x_len, y_len, _ = spatio_temporal_region.shape
        sptr_2d = spatio_temporal_region.as_2d
        distance_matrix = np.empty((x_len * y_len, x_len * y_len))

        # iterate each point in the region
        for i in range(0, x_len):
            for j in range(0, y_len):

                # for point (i, j), calculate the distances to all other points
                point = Point(i, j)
                self.logger.debug('Calculating distances at point: {}...'.format(str(point)))
                point_series = spatio_temporal_region.series_at(point)
                distances_at_point = [
                    self.measure(point_series, other_series)
                    for other_series
                    in sptr_2d
                ]
                self.logger.debug('Got: {}'.format(str(distances_at_point)))
                distance_matrix[i * y_len + j, :] = distances_at_point

        self.logger.debug('Distance matrix:')
        self.logger.debug(str(distance_matrix))

        return distance_matrix


class DistanceByRMSE(DistanceBetweenSeries):
    '''
    Use Root Mean Squared Error
    '''

    def measure(self, first_series, second_series):
        '''
        RMSE implementation
        '''
        # If a series is not defined (e.g. a forecast of a None model) then return NaN
        if np.isnan(first_series).any() or np.isnan(second_series).any():
            return np.nan
        else:
            return np.sqrt(np.square(np.subtract(first_series, second_series)).mean())

    def combine(self, distances_for_point):
        '''
        Uses the errors from each point to calculate a single RMSE.
        Note that the errors of each series can be added using Root Mean Squared.
        '''
        return arrays_util.root_mean_squared(distances_for_point)
