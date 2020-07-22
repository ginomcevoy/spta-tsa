'''
Module to calculate forecasting errors over spatio-temporal regions.
'''
import numpy as np

from .spatial import SpatialDecorator
from .function import FunctionRegionScalar
from .error_parallel import ParallelForecastError

from spta.util import arrays as arrays_util
from spta.util import error as error_util
from spta.util import log as log_util


class ErrorRegion(SpatialDecorator):
    '''
    A spatial region where each value represents the forecast error of a model.
    Uses the decorator pattern to allow integration with subclasses of SpatialRegion.

    Adds the overall_error property, which uses a supplied function to calculate a single
    value for the entire error value for the entire region (RMSE by default)

    Here we don't work with forecast/observation regions, for that see MeasureForecastingError.
    '''
    def __init__(self, decorated_region, error_combine_func=arrays_util.root_mean_squared,
                 **kwargs):
        super(ErrorRegion, self).__init__(decorated_region, **kwargs)
        self.error_combine_func = error_combine_func

    @property
    def overall_error(self):
        '''
        Calculate a single value for the forecast error in the region.
        By default, uses Root Mean Squared to find the RMSE value.
        '''
        error_list = []

        # this iterator will iterate over all the valid points in the region
        for point, error_at_point in self:
            error_list.append(error_at_point)

        # single value for region, e.g. RMSE
        # return arrays_util.root_mean_squared(error_list)
        return self.error_combine_func(error_list)

    def __next__(self):
        '''
        Use the decorated iteration, which may be more interesting than the default iteration
        from SpatialRegion.
        '''
        return self.decorated_region.__next__()


class MeasureForecastingError(FunctionRegionScalar):
    '''
    Given a forecast region, calculate an ErrorRegion where the error is calculated using an
    error function of the specified type (error_type).
    '''
    def __init__(self, error_func, observation_region, training_region=None):
        '''
        Create the function region that measures the error.

        error_func:
            the error function, obtained from get_error_func()

        observation_region:
            spatio-temporal region or subclass, always required

        training_region:
            spatio-temporal region or subclass, may be omitted if the error function does not
            require it (e.g. sMAPE).
        '''
        # we still need to use an internal numpy dataset to get stuff like region size
        super(MeasureForecastingError, self).__init__(observation_region.as_numpy)
        self.error_func = error_func

        # save the observation and training regions which are necessary to calculate errors
        # handle descaling here (early) so we don't have to handle at each point
        if observation_region.has_scaling():
            self.logger.debug('About to descale observation and training regions')
            self.observation_region = observation_region.descale()
            self.training_region = training_region.descale()
        else:
            self.logger.debug('MeasureForecastingError - No scaling detected')
            self.observation_region = observation_region
            self.training_region = training_region

        # self.logger.debug('Got observation region: {!r}'.format(observation_region))
        # self.logger.debug('Got training region: {!r}'.format(training_region))

    def function_at(self, point):
        '''
        Override the method that returns the function at each P(i, j), to calculate the error
        at P(i, j). Will use the observation series at P(i, j) and (if available) the training
        series at P(i, j). The inner function computes the error of the supplied forecast series.

        When this FunctionRegionScalar is applied to a forecast region, the forecast region is
        iterated (each point), and the inner function is called with the forecast series at each
        point. Note that we don't have access to the full forecast region in this scheme.
        '''

        # obtain observation and training series, last is optional
        observation_series = self.observation_region.series_at(point)
        training_series = None
        if self.training_region is not None:
            training_series = self.training_region.series_at(point)

        # the function that is evaluated over each point of the forecast region
        def forecast_error_at_point(forecast_series):

            # call the error function
            return self.error_func(forecast_series, observation_series, training_series)

        # the function as this point is the error function using data at this point
        return forecast_error_at_point

    def apply_to(self, forecast_region):
        '''
        Decorate the default apply_to implementation to return an ErrorRegion.
        '''

        # handle descaling here:
        forecast_region_ok = forecast_region
        if forecast_region.has_scaling():
            self.logger.debug('About to descale forecast region')
            forecast_region_ok = forecast_region.descale()

        spatial_region = super(MeasureForecastingError, self).apply_to(forecast_region_ok)
        return ErrorRegion(spatial_region)


class OverallErrorForEachForecast(FunctionRegionScalar):
    '''
    When applied to a forecast region, this function computes, for each point, the overall
    forecast error when repeating the forecast in that point to the entire region.

    For each point P(i,j), do:
        a) repeat the forecast series of P(i,j) over the entire observation region;
        b) compute an ErrorRegion with that created forecast region;
        c) save the overall error of that ErrorRegion as the return value at P(x,y).

    Example for a medoid: Take the forecasted series at the medoid, and compute the
    forecast error between that forecast series and each observation. Then compute the overall
    error by combining these errors, that is the resulting output value at the medoid.
    Repeat for all points in the observation region (x_len*y_len ErrorRegions are evaluated).
    '''

    def __init__(self, error_type, observation_region, training_region, parallel_workers=None):
        '''
        Prepare the overall error calculations. We need the error_type as string here,
        because we query another function factory that will support parallelization
        (overall_error_func instead of get_error_func)
        '''
        # we still need to use an internal numpy dataset to get stuff like region size
        super(OverallErrorForEachForecast, self).__init__(observation_region.as_numpy)

        # get a function that will calculate the overall error for a given forecast region
        self.overall_error_func = get_overall_error_func(error_type)

        # save the observation and training regions which are necessary to calculate errors
        self.observation_region = observation_region
        self.training_region = training_region

        # for parallelism
        self.parallel_workers = parallel_workers

    def function_at(self, point):
        '''
        Override the method that returns the function at each P(i, j).
        When this FunctionRegionScalar is applied to a forecast region, the forecast region is
        iterated (each point), and the inner function is called with the forecast series at each
        point. Note that we don't have access to the full forecast region in this scheme.
        '''

        self.logger.debug('OverallErrorForEachForecast at {}'.format(point))

        def overall_error_single_forecast(forecast_series):
            '''
            The function called at each point of this function region to find the forecast error
            of using a single forecast to predict the entire region.
            See overall_error_func for more information.
            '''
            # call top-level function (refactored out for parallelization)
            # don't do printing here, printing is for parallelization only
            overall_error_at_point = self.overall_error_func(point, forecast_series,
                                                             self.observation_region,
                                                             self.training_region,
                                                             with_print=False)

            self.logger.debug('Finished calculating error_single_model at {}'.format(point))

            # return the overall error as a single numerical value for the current forecast series
            return overall_error_at_point

        # the function that will be applied at each forecast point
        return overall_error_single_forecast

    def apply_to(self, forecast_region):
        '''
        Override the default application of this function region, so that it can support
        parallelism. When using parallelism, function_at will NOT be called!
        '''
        if self.parallel_workers:
            # Parallel implementation: use ParallelForecastError to parallelize the calculation
            # of ErrorRegions at each point.
            # notice that we will use one of the top-level functions defined below
            parallel_error = ParallelForecastError(self.parallel_workers, forecast_region,
                                                   self.observation_region, self.training_region)
            spatial_region = parallel_error.operate(self.overall_error_func)

        else:
            # use default serial behavior, which will call function_at(point)
            spatial_region = super(OverallErrorForEachForecast, self).apply_to(forecast_region)

        # this could be an Error Region but no need to get overall_error of overall_errors...
        return spatial_region


class ErrorAnalysis(log_util.LoggerMixin):
    '''
    Perform different types of calculations about forecasting over spatio-temporal regions,
    using both an observation region and a training region.

    For each point, measure the forecast error by using an observation series and an error
    function. The training series at that point may also be used for support, e.g. when using
    MASE as the error function.

    with_forecast_region:
        given a forecast region, use MeasureForecastingError function region to calculate an
        ErrorRegion, where the error at each point is calculated using an error function of the
        specified type (error_type).

    with_repeated_forecast:
        given a single forecast series, repeat this series over the entire observation region
        to compute a forecast region, then call with_forecast_region with it.

    overall_with_each_forecast:
        See OverallErrorForEachForecast for details.

    For with_forecast_region and overall_with_each_forecast, the entire forecast region is needed,
    which can be created by a model trained at each point of a training region.

    TODO normalized forecast should be denormalized before calculating prediction errors!?
    '''

    def __init__(self, observation_region, training_region=None, parallel_workers=None):
        '''
        Prepare the error analyis.

        observation_region:
            spatio-temporal region or subclass, always required

        training_region:
            spatio-temporal region or subclass, may be omitted if the error function does not
            require it (e.g. sMAPE).
        '''
        self.observation_region = observation_region
        self.training_region = training_region
        self.parallel_workers = parallel_workers

    def with_forecast_region(self, forecast_region, error_type):
        '''
        Given a forecast region, use MeasureForecastingError function region to calculate an
        ErrorRegion, where the error at each point is calculated using an error function.

        error_type:
            string that identifies the error function, see error_functions()
        '''
        error_func = get_error_func(error_type)
        measure_error = MeasureForecastingError(error_func, self.observation_region,
                                                self.training_region)
        error_region = measure_error.apply_to(forecast_region)
        return error_region

    def with_repeated_forecast(self, forecast_series, error_type):
        '''
        Given a single forecast series, repeat this series over the entire observation region
        to compute a forecast region, then call with_forecast_region with it.
        '''

        # Use the observation region to create a forecast region, where all the series are the
        # same forecast series repeated over the entire region. Using the observation region
        # is useful to keep subclass behavior, e.g. clusters.
        repeated_forecast_point = self.observation_region.repeat_series(forecast_series)
        error_region = self.with_forecast_region(repeated_forecast_point, error_type)
        return error_region

    def overall_with_each_forecast(self, forecast_region, error_type):
        '''
        See OverallErrorForEachForecast for details.
        '''

        # for each point, compute the overall error of using the repeated forecast at that point
        measure_overall_each_forecast = \
            OverallErrorForEachForecast(error_type, self.observation_region, self.training_region,
                                        self.parallel_workers)

        overall_error_region = measure_overall_each_forecast.apply_to(forecast_region)
        return overall_error_region


def error_functions():
    '''
    New error functions must be specified here.
    '''
    return {
        'MASE': error_util.mase,
        'sMAPE': error_util.smape
    }


def get_error_func(error_type):
    '''
    Choose an error function.
    '''
    # check availability of error_type
    if error_type not in error_functions().keys():
        raise ValueError('Error type not supported: {}'.format(error_type))

    return error_functions()[error_type]


def get_overall_error_func(error_type):
    '''
    Choose an overall error function. New error functions must be specified here and implemented
    below.

    Cannot rely on get_erorr_func only, because we need to pass a top-level function when using the
    parallel implementation for OverallErrorForEachForecast. For example, overall_error_mase
    will be passed to the parallel code in error_parallel.
    '''
    overall_error_functions = {
        'MASE': overall_error_mase,
        'sMAPE': overall_error_smape
    }

    # check availability of error_type
    if error_type not in overall_error_functions.keys():
        raise ValueError('Error type not supported: {}'.format(error_type))

    return overall_error_functions[error_type]


def overall_error_mase(point, forecast_series, observation_region, training_region,
                       with_print=True):
    '''
    Apply overall_error_generic using the MASE error.
    '''
    mase_func = get_error_func('MASE')
    return overall_error_generic(mase_func, point, forecast_series, observation_region,
                                 training_region, with_print)


def overall_error_smape(point, forecast_series, observation_region, training_region,
                        with_print=True):
    '''
    Apply overall_error_generic using the sMAPE error.
    '''
    smape_func = get_error_func('sMAPE')
    return overall_error_generic(smape_func, point, forecast_series, observation_region,
                                 training_region, with_print)


def overall_error_generic(error_func, point, forecast_series, observation_region, training_region,
                          with_print):
    '''
    See OverallErrorForEachForecast for details.
    '''
    # sanity check: no forecast, return nan
    if np.isnan(forecast_series).all():
        return np.nan

    # Use the observation region to create a forecast region, where all the series are the
    # same forecast series repeated over the entire region. Using the observation region
    # is useful to keep subclass behavior, e.g. clusters.
    repeated_forecast_series = observation_region.repeat_series(forecast_series)
    measure_error = MeasureForecastingError(error_func, observation_region, training_region)
    error_region = measure_error.apply_to(repeated_forecast_series)
    overall_error = error_region.overall_error

    # for parallel implementations, should be printed in a separate file
    # to avoid filling up the output, only print first in row
    if with_print and point.y == 0:
        print('overall_error {} for {} at {} = {:.3f}'.format(error_func.__name__,
                                                              observation_region,
                                                              point, overall_error), flush=True)
    return overall_error
