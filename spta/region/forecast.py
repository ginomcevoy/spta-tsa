import numpy as np

from spta.distance.dtw import DistanceByDTW
from spta.region import SpatioTemporalRegion
from spta.region.function import FunctionRegionScalar

from spta.util import log as log_util
from spta.util import arrays as arrays_util
from spta.util import error as error_util

from .spatial import SpatialDecorator


class ErrorRegion(SpatialDecorator):
    '''
    A spatial region where each value represents the forecast error of a model.
    Uses the decorator pattern to allow integration with new subclasses of SpatialRegion.

    Here we don't work with forecast/observation regions, subclasses will.
    '''
    def __init__(self, decorated_region, **kwargs):
        super(ErrorRegion, self).__init__(decorated_region, **kwargs)

    @property
    def overall_error(self):
        '''
        Calculate a single value for the forecast error in the region.
        We use Root Mean Squared to find the RMSE value.
        '''
        error_list = []

        # this iterator will iterate over all the valid points in the region
        for point, error_at_point in self:
            error_list.append(error_at_point)

        # single RMSE value for region
        return arrays_util.root_mean_squared(error_list)

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

    def point_with_max_error(self):
        '''
        Searches the errors in the region for the largest. Returns the (x, y) coordinates as
        region.Point (2d)

        Uses the instance iterator, cannot be used inside another iteration over itself!
        '''
        # save the max value
        max_value = -np.Inf
        max_point = None

        # use the iterator, should work as expected for subclasses of SpatialRegion
        for (point, value) in self:
            self.logger.debug('{} -> {}'.format(point, value))
            if value > max_value:
                max_value = value
                max_point = point

        return max_point

    def __next__(self):
        '''
        Use the decorated iteration, which may be more interesting than the default iteration
        from SpatialRegion.
        '''
        return self.decorated_region.__next__()


class ErrorRegionMASE(ErrorRegion):
    '''
    A spatial region where each value represents the forecast error of a model using MASE
    (Mean Absolute Scaled Error).

    In addition to observation values, MASE requires values in the training region (Yi) to scale
    the forecast error:

    qt = et / [(1 / n-1) * sum(| Y_i - Y_{i-1}|, i=2, i=n)
    MASE = mean(|qt|)
    '''
    # def __init__(self, decorated_region, **kwargs):
    #     super(ErrorRegionMASE, self).__init__(decorated_region, **kwargs)

    # def create_from_forecast(cls, forecast_region, observation_region, training_region):
    def __init__(self, forecast_region, observation_region, training_region):
        '''
        Create an instance of ErrorRegionMASE, by aplying MASE to each point in the forecast
        region. The instance will be region with the MASE error in each point.
        This requires not only the forecast and observation regions, but also the training region.
        '''
        (forecast_len, f_x_len, f_y_len) = forecast_region.shape
        (observation_len, o_x_len, o_y_len) = observation_region.shape
        (training_len, t_x_len, t_y_len) = training_region.shape

        # sanity check: all have the same 2D region
        assert (f_x_len, f_y_len) == (o_x_len, o_y_len)
        assert (f_x_len, f_y_len) == (t_x_len, t_y_len)

        # sanity check: forecasts and observations have the same length
        assert forecast_len == observation_len
        log_msg = 'Forecast: <{}> Observation: <{}>'
        self.logger.debug(log_msg.format(forecast_region.shape, observation_region.shape))

        # tell observation_region to create a new region, this way we get behavior from subclasses
        # of SpatialRegion
        decorated_region = observation_region.empty_region_2d()
        error_region_np = decorated_region.as_numpy

        # iterate over the observation points
        # note: we cannot iterate over forecast points, it breaks OverallErrorForEachForecast
        for (point_ij, observation_series_ij) in observation_region:

            # self.logger.debug('iterating observation region {}'.format(point_ij))

            # corresponding forecast and training series at Point(i, j)
            forecast_series_ij = forecast_region.series_at(point_ij)
            training_series_ij = training_region.series_at(point_ij)

            # calculate MASE for Point(i, j)
            error_ij = error_util.mase(forecast_series_ij, observation_series_ij,
                                       training_series_ij)
            error_region_np[point_ij.x, point_ij.y] = error_ij

        # finally create the instance of this class: the decorated region contains the data,
        # we wrap around this region to get ErrorRegion methods
        super(ErrorRegionMASE, self).__init__(decorated_region)


class OverallErrorForEachForecast(FunctionRegionScalar):
    '''
    When applied to a forecast region, this function computes, for each point, the overall
    foreecast error when repeating the forecast in that point to the entire region.

    Explanation: for point P(0, 0), take the forecasted series at P(0, 0) and compute the forecast
    error between that forecast series and each observation. Then compute the overall error by
    combining these errors, that is the resulting output value at P(0, 0). The same is done for
    each point.

    Example: the value of the result at the centroid will be the overall error computed, when
    using the forecasted series at the centroid to forecast the entire region.

    This function is meant to be applied to the output of ArimaModelRegion, which produces a
    forecast region where each point is forecasted by its own ARIMA model.
    '''

    def __init__(self, observation_region, training_region):
        # we still need to use an internal numpy dataset to get stuff like region size
        super(OverallErrorForEachForecast, self).__init__(observation_region.as_numpy)

        # save the observation and training regions which are necessary to calculate errors
        self.observation_region = observation_region
        self.training_region = training_region

    def function_at(self, point):
        '''
        Override the method that returns the function at each P(i, j).
        When this FunctionRegionScalar is applied to a forecast region, the forecast region is
        iterated (each point), and the inner function is called with the forecast series at each
        point. Note that we don't have access to the full forecast region in this scheme.
        '''

        self.logger.debug('OverallErrorForEachForecast at {}'.format(point))

        def error_single_model(forecast_series):
            '''
            The function called at each point of this function region to find the forecast error
            of using a single forecast to predict the entire region.
            See error_single_model_mase for more information.
            '''
            # call top-level function (refactored out for parallelization)
            # don't do printing here, printing is for parallelization only
            overall_error_at_point = error_single_model_mase(point, forecast_series,
                                                             self.observation_region,
                                                             self.training_region,
                                                             with_print=False)

            self.logger.debug('Finished calculating error_single_model at {}'.format(point))

            # return the overall error as a single numerical value for the current forecast series
            return overall_error_at_point

        # the function that will be applied at each forecast point
        return error_single_model


def error_single_model_mase(point, forecast_series, observation_region, training_region,
                            with_print=True):
    '''
    The function called at each point of this function region.
    The domain of the function region is the forecast region, and the forecast_series
    parameter here corresponds to the forecast at each point.

    We use this forecast to create a "fake" forecast region, where all values are the same
    as forecast_series, then calculate the overall error of that forecast as the return
    value of the function.

    This function is separated from OverallErrorForEachForecast to allow parallelization.
    '''

    # sanity check: no forecast, return nan
    if np.isnan(forecast_series).all():
        return np.nan

    # recover shape
    (_, x_len, y_len) = observation_region.shape

    # the "fake" forecast region where all points are the same as in the current point
    # we need a spatio-temporal region to store the repeated forecast data

    # This should work when using decorated regions, because:
    # 1. the iteration to call this error_single_model function is determined by the
    #   (possibly decorated) domain region.
    # 2. the ErrorRegionMASE below will iterate over the observation region, not
    #   the forecast region!
    # TODO: improve this hack?
    forecast_rep = SpatioTemporalRegion.repeat_series_over_region(forecast_series,
                                                                  (x_len, y_len))

    # error for the entire region using that forecast
    # use MASE to calculate the forecast error at each point
    error_region = ErrorRegionMASE(forecast_rep, observation_region, training_region)

    # return the overall error as a single numerical value for the current forecast series
    overall_error = error_region.overall_error

    # for parallel implementations, should be printed in a separate file
    # to avoid filling up the output, only print first in row
    if with_print and point.y == 0:
        print('error_single_model for {} at {} = {:.3f}'.format(observation_region,
                                                                point, overall_error), flush=True)

    return overall_error
