import numpy as np
from tslearn.metrics import dtw

from spta.util import arrays as arrays_util


class DistanceBetweenSeries:

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
