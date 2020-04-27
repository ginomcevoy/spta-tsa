import numpy as np

from spta.util import arrays as arrays_util
from . import DistanceBetweenSeries


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
