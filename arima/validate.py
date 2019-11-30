import numpy as np

from . import region
from . import util


def error_by_rmse(forecast_series, actual_series):
    return np.mean((forecast_series - actual_series)**2)**.5


class ErrorRegion(region.SpatialRegion):

    @property
    def combined_rmse(self):
        '''
        Uses the errors from each point to calculate a single RMSE.
        Note that the errors of each series can be added after squaring the RMSE
        (which has an outer square root).
        '''
        errors = self.as_array
        return np.sum(errors**2)**.5

    def point_with_least_error(self):
        '''
        Searches the errors in the region for the smallest. Returns the (x, y) coordinates as
        region.Point (2d)
        '''
        (minimum, index) = util.minimum_value_and_index(self.numpy_dataset)
        p = region.Point(index[0], index[1])
        print('Found minimum error %s at %s' % (minimum, p))
        return p

    @classmethod
    def create_from_forecasts(cls, forecast_region, test_region, error_func=error_by_rmse):

        (x1_len, y1_len, series1_len) = forecast_region.shape
        (x2_len, y2_len, series2_len) = test_region.shape

        # we need them to be about the same region and same series length
        assert((x1_len, y1_len, series1_len) == (x2_len, y2_len, series2_len))

        # work with lists, each element is a time series
        forecast_2d = forecast_region.as_list
        test_2d = test_region.as_list

        # Use list comprehension and zip to iterate over the two lists at the same time.
        # This will combine the forecast and test series of the same point, for each point.
        # We have a list of errors
        error_1d = [
            error_by_rmse(forecast_series, test_series)
            for (forecast_series, test_series)
            in zip(forecast_2d, test_2d)
        ]

        # recreate the region
        error_numpy_dataset = region.reshape_1d_to_2d(error_1d, x1_len, y1_len)
        return ErrorRegion(error_numpy_dataset)
