'''
Training of ARIMA models. Contains:

- train_arima, a wrapper for the ARIMA implementation
- ArimaTrainer, a function region that trains ARIMA models when applied to a training region.
'''
import functools
import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
# from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima

from spta.region.function import FunctionRegionScalarSame, FunctionRegionSeriesSame
from spta.util import log as log_util

from . import ArimaPDQ, forecast

# For auto_arima, the order is extracted from the model.
# Use this sentinel value when there is no model.
ORDER_WHEN_NO_MODEL = (-1, -1, -1)


def train_arima_pdq(arima_pdq, training_series):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.

    Returns a trained ARIMA model than can be used for forecasting (model fit).
    If the evaluation fails, return None instead of the model fit.
    '''
    logger = log_util.logger_for_me(train_arima_pdq)
    logger.debug('Training ARIMA with: %s' % str(arima_pdq))

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
        logger.warn('ARIMA {} failed with ValueError: {}'.format(arima_pdq, err))
        fitted_model = None
    except np.linalg.LinAlgError as err:
        # the "SVD did not converge" can create an error
        logger.warn('ARIMA {} failed with LinAlgError: {}'.format(arima_pdq, err))
        fitted_model = None

    return fitted_model


def train_auto_arima(auto_arima_params, training_series):
    '''
    run pyramid.arima.auto_arima to discover "optimal" p, d, q for a training_series, and fit the
    resulting model with the same training data.
    '''
    logger = log_util.logger_for_me(train_auto_arima)
    logger.debug('Running auto_arima with training size {}'.format(len(training_series)))

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
        p, d, q = sarimax_model.order
        arima_params = ArimaPDQ(p, d, q)
        fitted_model = train_arima_pdq(arima_params, training_series)

    except ValueError as err:
        logger.warn('ARIMA failed with ValueError: {}'.format(err))
        fitted_model = None

    except np.linalg.LinAlgError as err:
        # the "SVD did not converge" can create an error
        logger.warn('ARIMA failed with LinAlgError: {}'.format(err))
        fitted_model = None

    return fitted_model


class ArimaTrainer(FunctionRegionScalarSame):
    '''
    A FunctionRegion that uses the train_arima function to train an ARIMA model over a training
    region. Applying this FunctionRegion to the training spatio-temporal region will produce an
    instance of ArimaModelRegion (which is also a SpatialRegion). The shape of ArimaTrainer
    will be [x_len, y_len], where the training region has shape [train_len, x_len, y_len].

    The ArimaModelRegion is also a FunctionRegion, and it will contain a trained ARIMA model
    in each point P(i, j), trained with the training data at P(i, j) of the training region. The
    ArimaModelRegion can later be applied to another region to obtain the ForecastRegion
    (spatio-temporal region).

    To create an instance of this class by using a training region, use either:

        - with_hyperparameters():
            Create ARIMA models with the same supplied hyper-parameters in all region points.

        - with_auto_arima():
            Uses AutoARIMA to determine the hyper-parameters for each region point.

    class method. This will produce a different ARIMA model in each point.
    '''

    def __init__(self, arima_training_function, x_len, y_len):
        '''
        Initializes an instance of this function region, which is made of partial calls to
        train_arima(). Since the output of those calls is an object (an ARIMA model), the dtype
        needs to be set to object.
        '''
        super(ArimaTrainer, self).__init__(arima_training_function, x_len, y_len, dtype=object)

    def apply_to(self, training_region):
        '''
        Apply this function to train ARIMA models over a training region.

        The ouput of the parent behavior is to create a SpatialRegion, but we want to create an
        ArimaModelRegion instance. The SpatialRegion output already contains a trained model in
        each point of the training region, so this method is decorated to produce ArimaModelRegion.
        '''
        # get result from parent behavior
        spatial_region = super(ArimaTrainer, self).apply_to(training_region)

        # count and log missing models, iterate to find them
        self.missing_count = 0
        for (point, arima_model) in spatial_region:
            # if isinstance(arima_model, FailedArima):

            if arima_model is None:
                self.missing_count += 1

        if self.missing_count:
            self.logger.warn('Missing ARIMA models: {}'.format(self.missing_count))
        else:
            self.logger.info('ARIMA was trained in all points successfully.')

        # return ArimaModelRegion (a FunctionRegionSeries) instead of a plain SpatialRegion,
        # so that it can be applied to another region and produce forecasts over that region.
        # TODO: should ArimaModelRegion be a SpatialCluster?
        return forecast.ArimaModelRegion(spatial_region.as_numpy)

    @classmethod
    def with_hyperparameters(cls, arima_params, x_len, y_len):
        '''
        Creates an instance of this class, a function region. When applied to a training region,
        this function region will produce a different ARIMA model in each point of the training
        region, but all models will have the same hyper-parameters.

        arima_params
            ArimaPDQ (p, d, q) hyper-parameters

        '''
        # the function signature to be handled by regions can only receive a series
        # so we need to use partial here
        arima_with_params = functools.partial(train_arima_pdq, arima_params)
        return ArimaTrainer(arima_with_params, x_len, y_len)

    @classmethod
    def with_auto_arima(cls, auto_arima_params, x_len, y_len):
        '''
        Creates an instance of this class based on AutoARIMA. When applied to a training region,
        this function region will run pyramid.arima.auto_arima to discover "optimal" p, d, q
        hyperparameters for each point of the training region, and fit the resulting model.

        Note that we can reuse ArimaTrainer for both train_arima_pdq and train_auto_arima, *ONLY*
        because they create models with the same signature!

        auto_arima_params
            AutoArimaParams (start_p, start_q, max_p, max_q, d, stepwise)
        '''

        # the function signature to be handled by regions can only receive a series
        # so we need to use partial here
        auto_arima_with_params = functools.partial(train_auto_arima, auto_arima_params)
        return ArimaTrainer(auto_arima_with_params, x_len, y_len)


def extract_aic(fitted_arima_at_point):
    '''
    Extracts the aic value of an ARIMA model after it has been trained.
    '''
    # Problem: ArimaModelRegion is always a full spatial region, so it does not iterate
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
    applied to an ArimaModelRegion instance (output of applying ArimaTrainer to a training region).
    '''

    def __init__(self, x_len, y_len):
        # use extract_aic as the function to be applied at each point of an ArimaModelRegion
        # instance.
        super(ExtractAicFromArima, self).__init__(extract_aic, x_len, y_len)


def extract_pdq(fitted_arima_at_point):
    '''
    Extracts the (p, d, q) order and returns it as a series, so that it can be stored
    in the resulting SpatioTemporalRegion
    '''
    # Problem: ArimaModelRegion is always a full spatial region, so it does not iterate
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
        # use extract_pdq as the function to be applied at each point of an ArimaModelRegion
        # instance.

        # create a spatio-temporal region of integer values for the ARIMA order
        # integers are needed because they are passed to an ARIMA model
        super(ExtractPDQFromAutoArima, self).__init__(extract_pdq, x_len, y_len, dtype=np.int8)
