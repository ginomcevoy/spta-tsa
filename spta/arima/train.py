'''
Training of ARIMA models, see spta.model.base.ModelRegion and spta.model.train.ModelTrainer for details.
'''
import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
# from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima

from spta.region import Point
from spta.region.function import FunctionRegionScalarSame, FunctionRegionSeriesSame

from spta.model.train import ModelTrainer

from .model import ModelRegionArima
from . import ArimaPDQ

# For auto_arima, the order is extracted from the model.
# Use this sentinel value when there is no model.
ORDER_WHEN_NO_MODEL = (-1, -1, -1)


class TrainerArimaPDQ(ModelTrainer):
    '''
    A function region used to train ARIMA models. Apply this function to a training region
    (spatio-temporal region) to get an instance of ModelRegionArima, which can then be used
    to create ARIMA forecasts.

    Assumes that the ARIMA parameters (p, d, q) are constant for all points in the region.
    '''

    def __init__(self, arima_params, x_len, y_len):
        '''
        arima_params:
            ArimaParams namedtuple with (p, d, q) hyper-parameters

        x_len, y_len:
            Determines the size of the 2D region
        '''
        super(TrainerArimaPDQ, self).__init__(model_params_and_shape=(arima_params, x_len, y_len))

    def training_function(self, arima_pdq, training_series):
        '''
        Run ARIMA on a time series using order(p, d, q) -> hyper parameters

        ValueError: The computed initial AR coefficients are not stationary
        You should induce stationarity, choose a different model order, or you can
        pass your own start_params.

        Returns a trained ARIMA model than can be used for forecasting (model fit).
        If the evaluation fails, return None instead of the model fit.
        '''

        # sanity check: no parameters means no model, no data means no model
        # useful for sparse datasets or when refitting
        if arima_pdq is None or training_series is None:
            return None

        self.logger.debug('Training ARIMA {} with training size {}'.format(arima_pdq, len(training_series)))

        try:
            # for statsmodels.tsa.arima.model.ARIMA:
            p, d, q = arima_pdq.p, arima_pdq.d, arima_pdq.q
            arima_model = ARIMA(training_series,
                                order=(p, d, q),
                                seasonal_order=(0, 0, 0, 0))
            fitted_model = arima_model.fit()

            # for pmdarima.arima.ARIMA
            # arima_model = ARIMA(order=(arima_params.p, arima_params.d, arima_params.q),
            #                     suppress_warnings=True)
            # fitted_model = arima_model.fit(training_series, disp=0)

        except ValueError as err:
            self.logger.warn('ARIMA {} failed with ValueError: {}'.format(arima_pdq, err))
            fitted_model = None
        except np.linalg.LinAlgError as err:
            # the "SVD did not converge" can create an error
            self.logger.warn('ARIMA {} failed with LinAlgError: {}'.format(arima_pdq, err))
            fitted_model = None

        return fitted_model

    def create_model_region(self, numpy_model_array):
        arima_model_region = ModelRegionArima(numpy_model_array)

        # save the number of failed models... ugly but works
        arima_model_region.missing_count = self.missing_count

        # create a spatial region with AIC values and store it inside the arima_models object.
        extract_aic = ExtractAicFromArima(arima_model_region.x_len, arima_model_region.y_len)
        arima_model_region.aic_region = extract_aic.apply_to(arima_model_region)

        aic_0_0 = arima_model_region.aic_region.value_at(Point(0, 0))
        self.logger.debug('AIC at (0, 0) = {}'.format(aic_0_0))

        return arima_model_region


class TrainerAutoArima(ModelTrainer):
    '''
    A function region used to train ARIMA models using auto ARIMA grid search.
    Apply this function to a training region (spatio-temporal region) to get an instance of
    ModelRegionArima, which can then be used to create ARIMA forecasts.

    Assumes that the grid search parameters are constant for all points in the region.
    However, the ARIMA parameters (p, d, q) found by the grid search can vary from point to point.
    '''

    def __init__(self, auto_arima_params, x_len, y_len):
        '''
        arima_params:
            AutoArimaParams namedtuple with the grid search parameters to find
            the (p, d, q) hyper-parameters at each point of the region.

        x_len, y_len:
            Determines the size of the 2D region
        '''
        super(TrainerAutoArima, self).__init__(model_params_and_shape=(auto_arima_params, x_len, y_len))

        # used internally to create ARIMA models with the hyper-parameters
        # obtained from the auto ARIMA grid search
        self.arima_trainer = TrainerArimaPDQ(None, x_len, y_len)

    def training_function(self, auto_arima_params, training_series):
        '''
        run pyramid.arima.auto_arima to discover "optimal" p, d, q for a training_series, and fit the
        resulting model with the same training data.
        '''
        self.logger.debug('Running auto_arima with training size {}'.format(len(training_series)))

        # sanity check: no parameters means no model, no data means no model
        # useful for sparse datasets or when refitting
        if auto_arima_params is None or training_series is None:
            return None

        try:
            # find p, d, q with auto_arima, get a model
            sarimax_model = auto_arima(training_series,
                                       start_p=auto_arima_params.start_p,
                                       start_q=auto_arima_params.start_q,
                                       max_p=auto_arima_params.max_p,
                                       max_q=auto_arima_params.max_q,
                                       d=auto_arima_params.d,
                                       stepwise=auto_arima_params.stepwise,
                                       seasonal=False,
                                       suppress_warnings=True)

            # create a new ARIMA model based on the order obtained from auto_arima
            # do this because pmdarima gives us a SARIMAX with no seasonality, instead of ARIMA.
            # abusing the interface of arima_trainer a bit (not using it as a function region)
            p, d, q = sarimax_model.order
            arima_params = ArimaPDQ(p, d, q)
            fitted_model = self.arima_trainer.training_function(arima_params, training_series)

        except ValueError as err:
            self.logger.warn('ARIMA failed with ValueError: {}'.format(err))
            fitted_model = None

        except np.linalg.LinAlgError as err:
            # the "SVD did not converge" can create an error
            self.logger.warn('ARIMA failed with LinAlgError: {}'.format(err))
            fitted_model = None

        return fitted_model

    def create_model_region(self, numpy_model_array):
        arima_model_region = ModelRegionArima(numpy_model_array)

        # save the number of failed models... ugly but works
        arima_model_region.missing_count = self.missing_count

        # create a spatial region with AIC values and store it inside the arima_models object.
        extract_aic = ExtractAicFromArima(arima_model_region.x_len, arima_model_region.y_len)
        arima_model_region.aic_region = extract_aic.apply_to(arima_model_region)

        # create a spatio-temporal region with (p, d, q) values and store it inside arima_models.
        extract_pdq = ExtractPDQFromAutoArima(arima_model_region.x_len, arima_model_region.y_len)
        arima_model_region.pdq_region = extract_pdq.apply_to(arima_model_region, 3)

        aic_0_0 = arima_model_region.aic_region.value_at(Point(0, 0))
        self.logger.debug('AIC at (0, 0) = {}'.format(aic_0_0))

        pdq_0_0 = arima_model_region.pdq_region.series_at(Point(0, 0))
        self.logger.debug('(p, d, q) at (0, 0) = {}'.format(pdq_0_0))

        return arima_model_region

    def create_refitter(self, arima_model_region):
        '''
        For auto ARIMA, the refitter will use TrainerArimaPDQ, where the hyper-parameters are extracted
        from the model region.
        '''
        return TrainerRefitArima(arima_model_region)


class TrainerRefitArima(ModelTrainer):
    '''
    Re-fit an existing ARIMA model with new data, using the same hyper-parameters already present.
    The whole new data is used as training data, the original training data is discarded.

    Implemented by extracting the p, d, q values from a trained ARIMA model at each point
    (if present) and implementing a specific instance of ArimaTrainerGeneric that holds different
    p, d, q values at each point.
    '''

    def __init__(self, arima_model_region):

        # initialize an array with None values (no model) by default
        x_len, y_len = arima_model_region.shape
        numpy_params_array = np.full((x_len, y_len), None, dtype=object)

        model_count = 0

        for point, model_for_point in arima_model_region:

            # this function takes care of corner cases
            (p, d, q) = extract_pdq(model_for_point)

            if (p, d, q) != ORDER_WHEN_NO_MODEL:
                # valid hyper-parameters, save at position
                x, y = (point.x, point.y)
                numpy_params_array[x, y] = ArimaPDQ(p, d, q)
                model_count = model_count + 1

        # a SpatialRegion (!) that will store the hyper-parameters for each point.
        model_params_region = arima_model_region.new_spatial_region(numpy_params_array)

        # the hyper parameters vary at each point of the region, use the model_params_region
        # constructor parameter
        super(TrainerRefitArima, self).__init__(model_params_region=model_params_region)

        self.logger.debug('TrainerRefitArima instance: found {} (p,d,q)'.format(model_count))

        # used internally to create ARIMA models with the hyper-parameters
        # obtained from the auto ARIMA grid search
        self.arima_trainer = TrainerArimaPDQ(None, x_len, y_len)

    def training_function(self, arima_params, training_series):
        # use the ARIMA trainer with the extracted (p, d, q) values at the current point.
        # abusing the interface of arima_trainer a bit (not using it as a function region)
        return self.arima_trainer.training_function(arima_params, training_series)

    def create_model_region(self, numpy_model_array):
        # reuse the ARIMA trainer logic again,
        # but we need the missing_count from this instance
        self.arima_trainer.missing_count = self.missing_count
        return self.arima_trainer.create_model_region(numpy_model_array)


def extract_aic(fitted_arima_at_point):
    '''
    Extracts the aic value of an ARIMA model after it has been trained.
    '''
    # Problem: ModelRegionArima is always a full spatial region, so it does not iterate
    # like a cluster if it was generated for a cluster.
    # In points outside of the cluster, the 'value' (fitted_arima_at_point) is 0.
    # Also, if ARIMA training/fitting fails, the model is None.
    if fitted_arima_at_point == 0 or fitted_arima_at_point is None:
        return np.nan
    else:
        return fitted_arima_at_point.aic


class ExtractAicFromArima(FunctionRegionScalarSame):
    '''
    Get the AIC value that was obtained while fitting an ARIMA model. This function should be
    applied to an ModelRegionArima instance (output of applying ArimaTrainer to a training region).
    '''

    def __init__(self, x_len, y_len):
        # use extract_aic as the function to be applied at each point of an ModelRegionArima
        # instance.
        super(ExtractAicFromArima, self).__init__(extract_aic, x_len, y_len)


def extract_pdq(fitted_arima_at_point):
    '''
    Extracts the (p, d, q) order and returns it as a series, so that it can be stored
    in the resulting SpatioTemporalRegion
    '''
    # Problem: ModelRegionArima is always a full spatial region, so it does not iterate
    # like a cluster if it was generated for a cluster.
    # In points outside of the cluster, the 'value' (fitted_arima_at_point) is 0.
    # Also, if ARIMA training/fitting fails, the model is None.
    if fitted_arima_at_point == 0 or fitted_arima_at_point is None:
        (p, d, q) = ORDER_WHEN_NO_MODEL
    else:
        (p, d, q) = fitted_arima_at_point.model.order
    return np.array([p, d, q])


class ExtractPDQFromAutoArima(FunctionRegionSeriesSame):
    '''
    In the context of auto_arima analysis, extract the (p, d, q) order obtained, and store it
    as a... SpatioTemporalRegion?
    '''

    def __init__(self, x_len, y_len):
        # use extract_pdq as the function to be applied at each point of an ModelRegionArima
        # instance.

        # create a spatio-temporal region of integer values for the ARIMA order
        # integers are needed because they are passed to an ARIMA model
        super(ExtractPDQFromAutoArima, self).__init__(extract_pdq, x_len, y_len, dtype=np.int8)
