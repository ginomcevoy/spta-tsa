from collections import namedtuple
import numpy as np
import time

from spta.model.train import SplitTrainingAndTestLast
from spta.model.error import ErrorAnalysis

from spta.util import log as log_util


ForecastErrors = namedtuple('ForecastErrors', ('each', 'minimum', 'min_local', 'centroid', 'maximum'))


class ForecastAnalysis(log_util.LoggerMixin):
    '''
    Manages the training of ModelRegion instances, their forecasts and the corresponding forecast errors.
    Uses the Strategy pattern to work with different implementations of ModelTrainer.

    NOTE: this implementation assumes that each time series is split into training and
    test series, where:

    1. the length of the test series is equal to the length of the forecast series
    2. the training region has the same shape as spt_region (the trainer has the correct shape already)
    '''

    DEFAULT_FORECAST_LENGTH = 8

    def __init__(self, trainer, parallel_workers=None):
        self.parallel_workers = parallel_workers

        # strategy for training models
        self.trainer = trainer

        # created when training
        self.model_region = None
        self.error_analysis = None

        # created when calling forecast_at_each_point
        self.forecast_region_each = None

    def train_models(self, spt_region, test_len):
        '''
        Separate a spatio-temporal region in training region and observation region.
        Then, build one model for each point, using its training series, returning a subclass
        of ModelRegion that can be later used for forecasting.
        '''
        # save the region for future reference
        self.spt_region = spt_region

        # create training/test regions, where the test region has the same series length as the
        # expected forecast.
        splitter = SplitTrainingAndTestLast(test_len)
        (self.training_region, self.test_region) = splitter.split(spt_region)
        self.test_len = test_len

        # prepare the error analysis
        self.error_analysis = ErrorAnalysis(self.test_region, self.training_region,
                                            self.parallel_workers)

        # call the strategy
        self.logger.info('Training models using: {}'.format(self.trainer.__class__.__name__))
        self.model_region = self.trainer.apply_to(self.training_region)

        return self.model_region

    def forecast_at_each_point(self, forecast_len, error_type):
        '''
        Create a forecast region, using the trained model at each point to forecast the
        series at that point, with the given forecast length. Also, compute the forecast error
        using the test data as observation.

        Requires a string indicating the type of forecast error to be used.
        See spta.model.error.get_error_func for available error functions.

        Returns the forecast region, the error region and the time it took to compute the forecast.
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # prepare a forecast: create an empty region.
        # This region will control the iteration though, and it will be of the same subclass
        # as the training region (and of the spatio-temporal region), thereby supporting clusters.
        empty_region_2d = self.training_region.empty_region_2d()

        # use the trained models to forecast for their respective points, measure time
        time_forecast_start = time.time()
        forecast_region_each = self.model_region.apply_to(empty_region_2d, forecast_len)
        forecast_region_each.name = 'forecast_region_each'
        time_forecast_end = time.time()
        time_forecast = time_forecast_end - time_forecast_start

        # handle scaling of forecasted series:
        # if the models trained with scaled data, then the forecast will also be scaled
        # this does not undo the scaling, but it gives a chance to be descaled in error_analysis...
        # TODO improve approach to: a) cleaner, b) have access to both scaled/descaled?
        if self.spt_region.has_scaling():
            self.logger.debug('Adding scaling information to forecast_region_each')
            forecast_region_each = self.spt_region.new_spatio_temporal_region(forecast_region_each.numpy_dataset)

        # save the forecast region with each model
        self.forecast_region_each = forecast_region_each

        # calculate the error for this forecast region
        error_region_each = self.error_analysis.with_forecast_region(forecast_region_each,
                                                                     error_type)

        return forecast_region_each, error_region_each, time_forecast

    def forecast_whole_region_with_single_model(self, point, forecast_len, error_type):
        '''
        Using the model that was trained at the specified point, forecast the entire region
        with the specified forecast length, and calculate the error.

        When not using scaling, the forecast is the same for each point, since only depends on how
        the particular model was trained. With scaling however, the actual forecast will change:
        the scaled constant is the same, but the local scaling at the point will produce differences.

        Example: create the forecast and evaluate the error using the model that was trained
        at the centroid of the region, then replicate that forecast over the entire region, effectively
        using a single model (representative) for the entire region.

        See spta.model.error.get_error_func for available error functions.

        Returns the error region.
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # reuse the forecast at each point
        # if not available, calculate it now
        if self.forecast_region_each is None:
            self.forecast_at_each_point(forecast_len, error_type)

        # now we should have the forecast region
        # use the forecast series at the specified point, and do corresponding error analyis
        assert self.forecast_region_each is not None
        forecast_series = self.forecast_region_each.series_at(point)

        error_region_with_repeated_forecast = \
            self.error_analysis.with_repeated_forecast(forecast_series, error_type)

        return error_region_with_repeated_forecast

    def forecast_whole_region_with_all_models(self, forecast_len, error_type):
        '''
        Consider models at different points as representatives of the region. For a given
        point, use the forecast made by the model at *that* point, and use it to predict the
        observed values in the entire region. Obtain the MASE errors for each point, then compute
        the overall error made by combining the errors (RMSE).

        See see spta.model.error.OverallErrorForEachForecast for details.
        See spta.model.error.get_overall_error_func for available error functions.
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # reuse the forecast at each point
        # if not available, calculate it now
        # NOTE: this assumes same forecast length!
        if self.forecast_region_each is None:
            self.forecast_at_each_point(forecast_len, error_type)

        # now we should have the forecast region
        assert self.forecast_region_each is not None

        # call appropriate error analysis
        overall_error_region = \
            self.error_analysis.overall_with_each_forecast(self.forecast_region_each, error_type)

        return overall_error_region

    def analyze_errors(self, spt_region, error_type, forecast_len=DEFAULT_FORECAST_LENGTH):
        '''
        Perform the following analysis on a spatio-temporal region:

        1. Build one model for each point, forecast forecast_len days and compute the error
           of each forecast. This will create an error region (error_region_each).
           Combine the errors using distance_measure.combine to obtain a single metric for the
           prediction error. (error_region_each.combined_error)

        2. Consider models at different points as representatives of the region. For a given point,
           use the forecast made by the model at *that* point, and use it to predict the
           observed values in the entire region. Obtain the MASE errors for each point
           ("error_single_model"), then compute the overall error made by combining the errors (RMSE).

        3. Consider the following points for 2.:
            - minimum: the point that minimizes the error_single_model, i.e. has the minimum RMSE
                of the MASE errors on each point, when using its model to forecast the entire region.

            - min_local: the point that has the smallest forecast MASE error when forecasting its own
                observation.

            - centroid: the centroid of the region calculated externally, or using DTW if not provided.

            - maximum: the point that maximizes its error_single_model. This is the worst possible
                model for the region, used for comparison.

        The centroid should be provided in the spatio-temporal region as spt_region.centroid
        (it does not depend on the model, only on the dataset). If not provided, error is np.nan.
        '''

        # orchestrate the forecasting tasks above to achieve the requested results
        # time everything, this will be the compute time
        time_start = time.time()

        # train
        # NOTE: forecast_len = test_len!
        self.train_models(spt_region, forecast_len)

        # forecast using the model at each point
        each_result = self.forecast_at_each_point(forecast_len, error_type)
        (_, error_region_each, forecast_time) = each_result

        overall_error_each = error_region_each.overall_error
        log_msg = 'Combined {} error from all models: {}'
        self.logger.info(log_msg.format(error_type, overall_error_each))

        if spt_region.has_centroid:
            # show the local error at the centroid
            centroid = spt_region.centroid
            local_error_centroid = error_region_each.value_at(centroid)
            log_msg = 'Local error of model at the medoid {}: {}'
            self.logger.debug(log_msg.format(centroid, local_error_centroid))

        # find the errors when using each model to forecast the entire region
        overall_error_region = self.forecast_whole_region_with_all_models(forecast_len, error_type)

        # forecast computations finish here
        time_end = time.time()
        compute_time = time_end - time_start

        #
        # Best model: the one that minimizes the overall error when it is used to forecast
        # the entire region
        #
        (point_overall_error_min, overall_error_min) = overall_error_region.find_minimum()
        log_msg = 'Minimum overall {} error with single model at {}: {}'
        self.logger.info(log_msg.format(error_type, point_overall_error_min, overall_error_min))

        #
        # Find the model with minimum "local error"
        # This is the model that yields the minimum error when forecasting its own series.
        # The error computed is the error when using *that* model to forecast the entire region.
        #
        (point_min_local_error, _) = error_region_each.find_minimum()
        overall_error_min_local = overall_error_region.value_at(point_min_local_error)

        log_msg = '{} error from model with min local error at {}: {}'
        self.logger.info(log_msg.format(error_type, point_min_local_error,
                                        overall_error_min_local))

        #
        # Use the model at the centroid
        # ask the region for its centroid. If not available, the error is np.nan
        #
        if spt_region.has_centroid:
            centroid = spt_region.centroid
            overall_error_centroid = overall_error_region.value_at(centroid)
        else:
            centroid = None
            overall_error_centroid = np.nan
            self.logger.warn('Centroid was not pre-calculated!')

        log_msg = '{} error from ARIMA model at centroid {}: {}'
        self.logger.info(log_msg.format(error_type, centroid, overall_error_centroid))

        #
        # Worst ARIMA model: the one that maximizes the overall error when it is used to
        # to forecast the entire region
        #
        (point_overall_error_max, overall_error_max) = overall_error_region.find_maximum()
        log_msg = 'Maximum overall {} error with single ARIMA at {}: {}'
        self.logger.info(log_msg.format(error_type, point_overall_error_max, overall_error_max))

        # gather all errors
        overall_errors = ForecastErrors(overall_error_each, overall_error_min,
                                        overall_error_min_local, overall_error_centroid,
                                        overall_error_max)

        return overall_errors, forecast_time, compute_time

    def plot_distances_vs_errors(self, point_of_interest, forecast_len, error_type,
                                 distance_measure, plot_name=None, plot_desc=''):
        '''
        Use the model at a single point to create a forecast of the entire region, then plot the
        forecast errors against the distances of each point in the region to the specified point.
        Assumes that a distance matrix is present and obtainable via region metadata.
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # get the distances of each point in the region to the specified point
        # assumes that a distance matrix is present and obtainable via region metadata.
        distances_to_point = distance_measure.distances_to_point(self.spt_region,
                                                                 point_of_interest,
                                                                 self.spt_region.all_point_indices,
                                                                 use_distance_matrix=True)

        # use the model at the specified point to get a forecast for the entire region
        # and get the error
        error_region = self.forecast_whole_region_with_single_model(point_of_interest,
                                                                    forecast_len, error_type)

        # get the error values for the region, this also works on clusters
        forecast_errors = [
            forecast_error
            for _, forecast_error
            in error_region
        ]

        # convert to arrays and plot distances vs errors
        # TODO use plot_util.plot_distances_vs_forecast_errors
        from matplotlib import pyplot as plt
        _, subplot = plt.subplots(1, 1, figsize=(7, 5))
        subplot.plot(distances_to_point, forecast_errors, 'bo')

        # get ARIMA order at point of interest
        fitted_arima_at_point = self.model_region.value_at(point_of_interest)
        arima_order = fitted_arima_at_point.model.order

        # title
        title = 'Distances to medoid vs forecast errors at medoid'
        # title = title_str.format(self.spt_region, arima_order)
        subplot.set_title(title)

        # add some info about plot
        textstr = '\n'.join((
            '{}'.format(self.spt_region),
            'ARIMA: {}'.format(arima_order),
            '{}'.format(plot_desc)))
        subplot.text(0.05, 0.95, textstr, transform=subplot.transAxes,
                     verticalalignment='top')

        subplot.set_xlabel('DTW distances to medoid')
        subplot.set_ylabel('{} forecast errors'.format(error_type))
        subplot.grid(True, linestyle='--')

        if plot_name:
            plt.draw()
            plt.savefig(plot_name)
            self.logger.info('Saved figure: {}'.format(plot_name))
        plt.show()

    def check_forecast_request(self, forecast_len):
        '''
        Sanity checking for any forecast. Checks that the models have been trained, and that
        the error analysis is available.

        Also checks that the forecast length is equal to the test length that was used during
        training...
        '''
        assert forecast_len == self.test_len

        if self.model_region is None:
            raise ValueError('Forecast requested but models not trained!')

        if self.error_analysis is None:
            raise ValueError('ErrorAnalysis not available!')
