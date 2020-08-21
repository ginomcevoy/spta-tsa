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

from spta.region import Point
from spta.region.function import FunctionRegionScalar, FunctionRegionScalarSame, \
    FunctionRegionSeriesSame
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
    # sanity check: no parameters means no model, no data means no model
    # useful for sparse datasets or when refitting
    if arima_pdq is None or training_series is None:
        return None

    logger = log_util.logger_for_me(train_arima_pdq)
    logger.debug('Training ARIMA {} with training size {}'.format(arima_pdq, len(training_series)))

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


class ArimaTrainerGeneric(FunctionRegionScalar):
    '''
    A FunctionRegion that uses the train_arima function to train an ARIMA model over a training
    region. Applying this FunctionRegion to the training spatio-temporal region will produce an
    instance of ArimaModelRegion (which is also a SpatialRegion). The shape of ArimaTrainer
    will be [x_len, y_len], where the training region has shape [train_len, x_len, y_len].

    The resulting ArimaModelRegion (from applying this function region to a tranining region), is
    also a FunctionRegion, and it will contain a trained ARIMA model in each point P(i, j),
    trained with the training data at P(i, j) of the training region.

    The ArimaModelRegion can later be applied to another region to obtain an instance of
    ForecastRegion (spatio-temporal region).

    To create an instance of this class, see the create_from_hyper_params_matrix() class method.
    '''
    def __init__(self, matrix_of_partial_training_functions, **kwargs):
        '''
        Creates an instance of this class.

        matrix_of_partial_training_functions
            see create_from_hyper_params_matrix() class method
        '''
        super(ArimaTrainerGeneric, self).__init__(matrix_of_partial_training_functions,
                                                  dtype=object)

    def apply_to(self, training_region):
        '''
        Apply this function to train ARIMA models over a training region.

        The ouput of the parent behavior is to create a SpatialRegion, but we want to create an
        ArimaModelRegion instance. The SpatialRegion output already contains a trained model in
        each point of the training region, so this method is decorated to produce ArimaModelRegion.
        '''
        # get result from parent behavior
        # this will already call the training function and return trained ARIMA models.
        spatial_region = super(ArimaTrainerGeneric, self).apply_to(training_region)

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
    def create_from_hyper_params_matrix(cls, hyper_params_matrix, training_function):
        '''
        Initializes an instance of ArimaTrainerGeneric based on a matrix of hyper-parameters
        (a tuple of hyper-parameters for each point) and one of the available training functions,
        either train_arima_pdq or train_auto_arima.

        hyper_params_matrix
            numpy array with shape (x_len, y_len), each point should have a tuple with
            the hyper-parameters.

        training_function
            Either train_arima_pdq (if hyper-parameters are type ArimaPDQ) or train_auto_arima
            (if hyper-parameters are type AutoArimaParams)
        '''
        x_len, y_len = hyper_params_matrix.shape

        # For each point P, we need a function that can be applied to the corresponding series.
        # This function is constructed using the hyper-parameters at P and the training function.
        matrix_of_partial_training_functions = np.empty((x_len, y_len), dtype=object)
        for x in range(x_len):
            for y in range(y_len):
                matrix_of_partial_training_functions[x][y] = \
                    functools.partial(training_function, hyper_params_matrix[x][y])

        return ArimaTrainerGeneric(matrix_of_partial_training_functions)


class ArimaTrainer(ArimaTrainerGeneric):
    '''
    Prepares the training of ARIMA models in a (x_len, y_len) region, with the specified
    ARIMA hyper-parameters (instance of ArimaPDQ).

    When applied to a training region, this function region will produce a different ARIMA model
    in each point of the training region, but all models will have the same hyper-parameters.

    See ArimaTrainerGeneric for details.
    '''

    def __init__(self, arima_params, x_len, y_len):
        '''
        Creates an instance of this function region.

        arima_params
            instance of ArimaPDQ

        x_len
            size of region in x axis

        y_len
            size of region in y axis
        '''

        # Need to create the matrix_of_partial_training_functions argument for parent,
        # similar to create_from_hyper_params_matrix() implementation but more direct since
        # the hyper-parameters are constant.
        array_of_partial_training_functions = [
            functools.partial(train_arima_pdq, arima_params)
            for i in range(x_len * y_len)
        ]
        matrix_of_partial_training_functions = \
            np.array(array_of_partial_training_functions).reshape(x_len, y_len)

        # call parent
        super(ArimaTrainer, self).__init__(matrix_of_partial_training_functions)


class AutoArimaTrainer(ArimaTrainerGeneric):
    '''
    Prepares the training of autoARIMA models in a (x_len, y_len) region, with the specified
    autoARIMA hyper-parameters (instance of AutoArimaParams).

    When applied to a training region, this function region will run pyramid.arima.auto_arima to
    discover "optimal" p, d, q hyperparameters for each point of the training region, and fit the
    resulting model.

    See ArimaTrainerGeneric for details.
    '''
    def __init__(self, auto_arima_params, x_len, y_len):
        '''
        Creates an instance of this function region.

        auto_arima_params
            instance of AutoArimaParams

        x_len
            size of region in x axis

        y_len
            size of region in y axis
        '''

        # Need to create the matrix_of_partial_training_functions argument for parent,
        # similar to create_from_hyper_params_matrix() implementation but more direct since
        # the hyper-parameters are constant.
        array_of_partial_training_functions = [
            functools.partial(train_auto_arima, auto_arima_params)
            for i in range(x_len * y_len)
        ]
        matrix_of_partial_training_functions = \
            np.array(array_of_partial_training_functions).reshape(x_len, y_len)

        # call parent
        super(AutoArimaTrainer, self).__init__(matrix_of_partial_training_functions)


def refit_arima(arima_model_region, new_training_region):
    '''
    Re-fit an existing ARIMA model with new data, using the same hyper-parameters already present.
    The whole new data is used as training data, the original training data is discarded.

    Implemented by extracting the p, d, q values from a trained ARIMA model at each point
    (if present) and implementing a specific instance of ArimaTrainerGeneric that holds different
    p, d, q values at each point.
    '''
    # initialize an array with None values (no model) by default
    x_len, y_len = arima_model_region.shape
    hyper_params_matrix = np.full((x_len, y_len), None, dtype=object)

    # done per point because condition needs to be checked
    for x in range(0, x_len):
        for y in range(0, y_len):

            # this function takes care of corner cases
            (p, d, q) = extract_pdq(arima_model_region.value_at(Point(x, y)))

            if (p, d, q) != ORDER_WHEN_NO_MODEL:
                # valid hyper-parameters, save at position
                hyper_params_matrix[x, y] = ArimaPDQ(p, d, q)

    # this is similar to ArimaTrainer but from a matrix of hyper-parameters
    new_arima_trainer = ArimaTrainerGeneric.create_from_hyper_params_matrix(hyper_params_matrix,
                                                                            train_arima_pdq)
    return new_arima_trainer.apply_to(new_training_region)


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
