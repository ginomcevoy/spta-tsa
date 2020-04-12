from spta.util import arrays as arrays_util

from .spatial import SpatialRegion

from . import Point, reshape_1d_to_2d
from . import distance


class ErrorRegion(SpatialRegion):

    def __init__(self, numpy_dataset, distance_error):
        super(ErrorRegion, self).__init__(numpy_dataset)
        self.distance_error = distance_error

    @property
    def combined_error(self):
        errors = self.as_array
        return self.distance_error.combine(errors)

    def point_with_min_error(self):
        '''
        Searches the errors in the region for the smallest. Returns the (x, y) coordinates as
        region.Point (2d)
        '''
        (minimum, index) = arrays_util.minimum_value_and_index(self.numpy_dataset)
        p = Point(index[0], index[1])
        #  print('Found minimum error %s at %s' % (minimum, p))
        return p

    @classmethod
    def create_from_forecasts(cls, forecast_region, test_region,
                              distance_error=distance.DistanceByDTW()):

        (x1_len, y1_len, series1_len) = forecast_region.shape
        (x2_len, y2_len, series2_len) = test_region.shape

        # we need them to be about the same region and same series length
        assert((x1_len, y1_len, series1_len) == (x2_len, y2_len, series2_len))

        # work with lists, each element is a time series
        forecast_as_list = forecast_region.as_list
        test_as_list = test_region.as_list

        # Use list comprehension and zip to iterate over the two lists at the same time.
        # This will combine the forecast and test series of the same point, for each point.
        error_list = [
            distance_error.measure(forecast_series, test_series)
            for (forecast_series, test_series)
            in zip(forecast_as_list, test_as_list)
        ]

        # recreate the region
        error_numpy_dataset = reshape_1d_to_2d(error_list, x1_len, y1_len)
        return ErrorRegion(error_numpy_dataset, distance_error)
