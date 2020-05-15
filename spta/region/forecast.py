from spta.distance.dtw import DistanceByDTW

from spta.util import arrays as arrays_util
from spta.util import log as log_util

from .spatial import SpatialRegion

from . import Point, reshape_1d_to_2d


class ForecastLengthRegion(SpatialRegion):
    '''
    A spatial region [x_len, y_len] with a constant value on all points, representing the size of
    a forecasted series, forecast_len.

    This region can be applied to FunctionRegionSeries subclasses that represent forecasting
    models, so that the result would be SpatioTemporalRegion with shape
    [forecast_len, x_len, y_len]

    Not currently used by ARIMA.
    '''
    def __init__(self, numpy_dataset, forecast_len):
        super(ForecastLengthRegion, self).__init__(numpy_dataset)

        # save value for convenience
        self.forecast_len = forecast_len

    @classmethod
    def from_value_and_region(cls, forecast_len, sp_region):

        # get desired shape
        x_len, y_len = (sp_region.x_len, sp_region.y_len)

        # all elements are the same
        np_array = arrays_util.copy_value_as_matrix_elements(forecast_len, x_len, y_len)
        return ForecastLengthRegion(np_array, forecast_len)


class ErrorRegion(SpatialRegion):
    '''
    A spatial region where each value represents the forecast error of a model.
    It is created by measuring the distance between a forecast region and a test region.

    TODO reimplement this using FunctionRegionScalar!
    '''

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
                              distance_error=DistanceByDTW()):

        (series1_len, x1_len, y1_len) = forecast_region.shape
        (series2_len, x2_len, y2_len) = test_region.shape

        logger = log_util.logger_for_me(cls.create_from_forecasts)
        logger.debug('Forecast: <{}> Test: <{}>'.format(forecast_region.shape, test_region.shape))

        # we need them to be about the same region and same series length
        assert((series1_len, x1_len, y1_len) == (series2_len, x2_len, y2_len))

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
