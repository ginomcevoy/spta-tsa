import numpy as np
from spta.distance.dtw import DistanceByDTW
from spta.util import log as log_util

from .spatial import SpatialRegionDecorator


class ErrorRegion(SpatialRegionDecorator):
    '''
    A spatial region where each value represents the forecast error of a model.
    It is created by measuring the distance between a forecast region and a test region.

    Uses the decorator pattern to allow integration with new subclasses of SpatialRegion.
    '''

    def __init__(self, decorated_region, distance_measure, **kwargs):
        super(ErrorRegion, self).__init__(decorated_region, **kwargs)
        self.distance_measure = distance_measure

    @property
    def combined_error(self):

        # iterate over all elements to get error array, then combine the errors
        errors = []
        for point, error_at_point in self:
            errors.append(error_at_point)
        return self.distance_measure.combine(errors)

    def point_with_min_error(self):
        '''
        Searches the errors in the region for the smallest. Returns the (x, y) coordinates as
        region.Point (2d)

        Uses the instance iterator, cannot be used inside another iteration over itself!
        '''
        # save the min value
        min_value = np.Inf
        min_point = None

        # use the iterator, should work as expected for subclasses of SpatialRegion
        for (point, value) in self:
            if value < min_value:
                min_value = value
                min_point = point

        return min_point

    @classmethod
    def create_from_forecasts(cls, forecast_region, test_region,
                              distance_measure=DistanceByDTW()):

        (series1_len, x1_len, y1_len) = forecast_region.shape
        (series2_len, x2_len, y2_len) = test_region.shape

        logger = log_util.logger_for_me(cls.create_from_forecasts)
        logger.debug('Forecast: <{}> Test: <{}>'.format(forecast_region.shape, test_region.shape))

        # we need them to be about the same region and same series length
        assert((series1_len, x1_len, y1_len) == (series2_len, x2_len, y2_len))

        # # work with lists, each element is a time series
        # forecast_as_list = forecast_region.as_list
        # test_as_list = test_region.as_list

        # # Use list comprehension and zip to iterate over the two lists at the same time.
        # # This will combine the forecast and test series of the same point, for each point.
        # error_list = [
        #     distance_measure.measure(forecast_series, test_series)
        #     for (forecast_series, test_series)
        #     in zip(forecast_as_list, test_as_list)
        # ]

        # # recreate the region
        # error_numpy_dataset = reshape_1d_to_2d(error_list, x1_len, y1_len)
        # return ErrorRegion(error_numpy_dataset, distance_measure)

        # create SpatialRegion now, this will work when using a subclass too
        decorated_region = forecast_region.empty_region_2d()
        error_region_np = decorated_region.as_numpy

        # iterate over the forecast points
        for (point_ij, forecast_series_ij) in forecast_region:

            # corresponding test series
            test_series_ij = test_region.series_at(point_ij)

            # calculate the forecast error for this point
            error_ij = distance_measure.measure(forecast_series_ij, test_series_ij)
            error_region_np[point_ij.x, point_ij.y] = error_ij

        # wrap the data around the ErrorRegion decorator to get its methods
        return ErrorRegion(decorated_region, distance_measure)
