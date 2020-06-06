import numpy as np
from tslearn.metrics import dtw

from spta.region import Point
from spta.util import arrays as arrays_util

from . import DistanceBetweenSeries


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
            self.logger.debug('Calculating all distances at: {}...'.format(i))
            distances_for_i = [
                self.measure(series_i, other_series)
                for other_series
                in X
            ]
            self.logger.debug('Got: {}'.format(str(distances_for_i)))
            distance_matrix[i, :] = distances_for_i

        return distance_matrix

    def compute_distance_matrix_sptr(self, spatio_temporal_region):

        _, x_len, y_len = spatio_temporal_region.shape
        sptr_2d = spatio_temporal_region.as_2d
        distance_matrix = np.empty((x_len * y_len, x_len * y_len))

        # iterate each point in the region
        for i in range(0, x_len):
            for j in range(0, y_len):

                # for point (i, j), calculate the distances to all other points
                point = Point(i, j)
                self.logger.debug('DTW at: {}...'.format(str(point)))
                point_series = spatio_temporal_region.series_at(point)
                distances_at_point = [
                    self.measure(point_series, other_series)
                    for other_series
                    in sptr_2d
                ]
                # self.logger.debug('Got: {}'.format(str(distances_at_point)))
                distance_matrix[i * y_len + j, :] = distances_at_point

        # self.logger.debug('Distance matrix:')
        self.logger.debug(str(distance_matrix))

        return distance_matrix


class DistanceBySpatialDTW(DistanceByDTW):
    '''
    A DTW implementation that adds the euclidian distance between points as a weight to the
    DTW distance. The 'weight' parameter is used as an exponential of the euclidian distance, and
    the value is multiplied to the DTW distance.

    A weight of 0 is equivalent to DistanceByDTW.

    Only supported during computation of the distance matrix!
    Only supports spatio temporal data!
    '''

    def __init__(self, weight):
        super(DistanceBySpatialDTW, self).__init__()
        self.weight = weight
        self.weighted = False

    def weight_distance_matrix(self, region):
        '''
        Uses a pre-calculated distance matrix using DTW, but then adds the weigths for the
        spatio-temporal data.
        '''
        # matrix must be precomputed
        if self.distance_matrix is None:
            raise ValueError('Compute matrix before calling weight_distance_matrix')

        # don't add weight again...
        if self.weighted:
            return self.distance_matrix

        x_len, y_len = (region.x2 - region.x1, region.y2 - region.y1)

        # coordinates of the spatial reigon
        points_of_2d_region = arrays_util.list_of_2d_points(x_len, y_len)

        # iterate points
        for index in range(0, x_len * y_len):

            # recover 2d position of point
            point_at_index = [int(index / y_len), index % y_len]

            # euclidian distances to other points
            euclidians_to_point = np.linalg.norm(points_of_2d_region - point_at_index, axis=1)

            # update value of the distances
            self.distance_matrix[index] = self.distance_matrix[index] + \
                euclidians_to_point * self.weight
            # np.power(euclidians_to_point, self.weight)

        # flag
        self.weighted = True

        return self.distance_matrix

    def load_distance_matrix_2d(self, filename, expected_region):
        '''
        Loads a pre-computed DTW distance matrix from a file for a 2d region.
        THEN adds the weight of the euclidian distances to it.
        The distance matrix is expected to be a 2d matrix [x_len * y_len, x_len * y_len].
        '''

        # read normally
        super(DistanceBySpatialDTW, self).load_distance_matrix_2d(filename, expected_region)

        # add the weight
        return self.weight_distance_matrix(expected_region)


if __name__ == '__main__':

    # test loading distance matrix for SP_RJ
    from spta.dataset import sp_rj

    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_2d(sp_rj.SP_RJ_DISTANCES, sp_rj.SP_RJ_REGION)
    print('read distance matrix SP_RJ: {}'.format(distance_measure.distance_matrix.shape))
