'''
Forecasting with ARIMA models. Contains:

- ArimaModelRegion, a function region that holds a trained ARIMA model at each point. Can create
    a forecast region when called with an empty region.

- ArimaForecasting, a class that orchestrates the training and forecasting of ARIMA models.
'''
import time

from spta.region import Point
from spta.region.forecast import ForecastModelRegion
from spta.region.train import SplitTrainingAndTestLast
from spta.region.error import ErrorAnalysis

from spta.util import log as log_util

from .training import ArimaTrainer, ExtractAicFromArima, ExtractPDQFromAutoArima


class ArimaModelRegion(ForecastModelRegion):
    '''
    A FunctionRegion that uses the arima_forecast function to create a forecast region using
    ARIMA models.

    See ForecastModelRegion for more details.
    '''

    def forecast_from_model(self, model_at_point, forecast_len, value_at_point, point):
        '''
        Creates a forecast from a trained ARIMA model.

        When using statsmodels.tsa.arima_model.ARIMA:
            return model_at_point.forecast(forecast_len)[0]

        When using statsmodels.tsa.arima.model.ARIMA:
            return model_at_point.forecast(forecast_len)

        When using pmdarima.arima.ARIMA:
            return model_at_point.predict(forecast_len)
        '''
        return model_at_point.forecast(forecast_len)


class ArimaForecasting(log_util.LoggerMixin):
    '''
    Manages the training and forecasting of ARIMA models.
    Base class for using both specified p,d,q hyperparameters and auto_arima.
    '''

    def __init__(self, arima_params_obj, parallel_workers=None):
        self.arima_params_obj = arima_params_obj
        self.parallel_workers = parallel_workers

        # created when training
        self.arima_models = None
        self.error_analysis = None

        # created when calling forecast_at_each_point
        self.forecast_region_each = None

    def train_models(self, spt_region, test_len):
        '''
        Separate a spatio-temporal region in training region and observation region.
        Then, build one ARIMA model for each point, using its training series.

        Returns a spatial region that contains, for each point an ARIMA model that can be
        later used for forecasting.

        NOTE: this implementation assumes that each time series is split into training and
        test series, where the length of the test series is equal to the length of the forecast
        series!
        '''
        # save the region for future reference
        self.spt_region = spt_region

        # create training/test regions, where the test region has the same series length as the
        # expected forecast.
        splitter = SplitTrainingAndTestLast(test_len)
        (self.training_region, self.test_region) = splitter.split(spt_region)
        self.test_len = test_len

        # call specific ARIMA training implementation
        self.arima_models = self.train_models_impl(self.training_region, self.arima_params_obj)

        # prepare the error analysis
        self.error_analysis = ErrorAnalysis(self.test_region, self.training_region,
                                            self.parallel_workers)

        return self.arima_models

    def train_models_impl(self, training_region, arima_params_obj):
        '''
        Subclasses must specify how ARIMA models are trained, e.g. using p,d,q or auto_arima.
        '''
        raise NotImplementedError

    def forecast_at_each_point(self, forecast_len, error_type):
        '''
        Create a forecast region, using the trained ARIMA model at each point to forecast the
        series at that point, with the given forecast length. Also, compute the forecast error
        using the test data as observation.

        Requires a string indicating the type of forecast error to be used.
        See spta.region.error.get_error_func for available error functions.

        Returns the forecast region, the error region and the time it took to compute the forecast.

        NOTE: this implementation assumes that each time series is split into training and
        test series, where the length of the test series is equal to the length of the forecast
        series!
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # prepare a forecast: create an empty region.
        # This region will control the iteration though, and it will be of the same subclass
        # as the training region (and of the spatio-temporal region), thereby supporting clusters.
        empty_region_2d = self.training_region.empty_region_2d()

        # use the ARIMA models to forecast for their respective points, measure time
        # TODO put the timing code inside arima_models
        time_forecast_start = time.time()
        forecast_region_each = self.arima_models.apply_to(empty_region_2d, forecast_len)
        forecast_region_each.name = 'forecast_region_each'
        time_forecast_end = time.time()
        time_forecast = time_forecast_end - time_forecast_start

        # save the forecast region with each model
        self.forecast_region_each = forecast_region_each

        # calculate the error for this forecast region
        error_region_each = self.error_analysis.with_forecast_region(forecast_region_each,
                                                                     error_type)

        return forecast_region_each, error_region_each, time_forecast

    def forecast_whole_region_with_single_model(self, point, forecast_len, error_type):
        '''
        Using the ARIMA model that was trained at the specified point, forecast the entire region
        with the specified forecast length, and calculate the error.

        The forecast is the same for each point, since only depends on how the particular model
        was trained.

        Example: create the forecast and evaluate the error using the ARIMA model that was trained
        at the cetroid, and then replicate that forecast over the entire region, effectively using
        a single model for the entire region.

        See spta.region.error.get_error_func for available error functions.

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
        Consider ARIMA models at different points as representatives of the region. For a given
        point, use the forecast made by the model at *that* point, and use it to predict the
        observed values in the entire region. Obtain the MASE errors for each point, then compute
        the overall error made by combining the errors (RMSE).

        See see spta.region.error.OverallErrorForEachForecast for details.
        See spta.region.error.get_overall_error_func for available error functions.
        '''
        # check conditions
        self.check_forecast_request(forecast_len)

        # reuse the forecast at each point
        # if not available, calculate it now
        # TODO: this assumes same forecast length!
        if self.forecast_region_each is None:
            self.forecast_at_each_point(forecast_len, error_type)

        # now we should have the forecast region
        assert self.forecast_region_each is not None

        # call appropriate error analysis
        overall_error_region = \
            self.error_analysis.overall_with_each_forecast(self.forecast_region_each, error_type)

        return overall_error_region

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
        from matplotlib import pyplot as plt
        _, subplot = plt.subplots(1, 1, figsize=(7, 5))
        subplot.plot(distances_to_point, forecast_errors, 'bo')

        # get ARIMA order at point of interest
        fitted_arima_at_point = self.arima_models.value_at(point_of_interest)
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

        if self.arima_models is None:
            raise ValueError('Forecast requested but models not trained!')

        if self.error_analysis is None:
            raise ValueError('ErrorAnalysis not available!')


class ArimaForecastingPDQ(ArimaForecasting):
    '''
    Manages the training and forecasting of ARIMA models when using p, d, q hyperparameters.
    See ArimaForecasting for full implementation.
    '''

    def train_models_impl(self, training_region, arima_hyperparams):
        '''
        Train ARIMA models where the same (p, d, q) hyperparameters are used over the entire
        training region.
        '''
        self.logger.info('Using (p, d, q) = {}'.format(arima_hyperparams))

        _, x_len, y_len = training_region.shape

        # a function region with produces trained models when applied to a training region
        # using p, d, q
        arima_trainers = ArimaTrainer.with_hyperparameters(arima_hyperparams, x_len, y_len)

        # train the models: this returns an instance of ArimaModelRegion, that has an instance of
        # statsmodels.tsa.arima.model.ARIMAResults at each point
        arima_models = arima_trainers.apply_to(training_region)

        # save the number of failed models... ugly but works
        arima_models.missing_count = arima_trainers.missing_count

        # create a spatial region with AIC values and store it inside the arima_models object.
        extract_aic = ExtractAicFromArima(x_len, y_len)
        arima_models.aic_region = extract_aic.apply_to(arima_models)

        aic_0_0 = arima_models.aic_region.value_at(Point(0, 0))
        self.logger.debug('AIC at (0, 0) = {}'.format(aic_0_0))

        return arima_models


class ArimaForecastingAutoArima(ArimaForecasting):
    '''
    Manages the training and forecasting of ARIMA models when using auto_arima.
    See ArimaForecasting for full implementation.
    '''

    def train_models_impl(self, training_region, auto_arima_params):
        '''
        Train ARIMA models using auto_arima, which may choose different (p, d, q) hyperparameters
        throughout the training region.
        '''
        _, x_len, y_len = training_region.shape

        # a function region with produces trained models when applied to a training region
        # using auto_arima
        arima_trainers = ArimaTrainer.with_auto_arima(auto_arima_params, x_len, y_len)

        # train the models: this returns an instance of ArimaModelRegion, that has an instance of
        # statsmodels.tsa.arima.model.ARIMAResults at each point
        # this is because we used auto_arima to get (p, d, q) order and then trained
        # statsmodels.tsa.model.arima.ARIMA model with it
        arima_models = arima_trainers.apply_to(training_region)

        # save the number of failed models... ugly but works
        arima_models.missing_count = arima_trainers.missing_count

        # create a spatial region with AIC values and store it inside the arima_models object.
        extract_aic = ExtractAicFromArima(x_len, y_len)
        arima_models.aic_region = extract_aic.apply_to(arima_models)

        # create a spatio-temporal region with (p, d, q) values and store it inside arima_models.
        extract_pdq = ExtractPDQFromAutoArima(x_len, y_len)
        arima_models.pdq_region = extract_pdq.apply_to(arima_models, 3)

        aic_0_0 = arima_models.aic_region.value_at(Point(0, 0))
        self.logger.debug('AIC at (0, 0) = {}'.format(aic_0_0))

        pdq_0_0 = arima_models.pdq_region.series_at(Point(0, 0))
        self.logger.debug('(p, d, q) at (0, 0) = {}'.format(pdq_0_0))

        return arima_models
