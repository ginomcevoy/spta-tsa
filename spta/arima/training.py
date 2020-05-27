'''
Training of ARIMA models. Contains:

- train_arima, a wrapper for the ARIMA implementation
- ArimaTrainer, a function region that trains ARIMA models when applied to a training region.
'''
import functools
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

from spta.region.function import FunctionRegionScalar

from spta.util import arrays as arrays_util
from spta.util import log as log_util

from . import forecast


def train_arima(arima_params, time_series):
    '''
    Run ARIMA on a time series using order(p, d, q) -> hyper parameters

    ValueError: The computed initial AR coefficients are not stationary
    You should induce stationarity, choose a different model order, or you can
    pass your own start_params.

    Returns a trained ARIMA model than can be used for forecasting (model fit).
    If the evaluation fails, return None instead of the model fit.
    '''
    log = log_util.logger_for_me(train_arima)
    log.debug('Training ARIMA with: %s' % str(arima_params))

    try:
        model = ARIMA(time_series, order=(arima_params.p, arima_params.d, arima_params.q))
        model_fit = model.fit(disp=0)
    except ValueError as err:
        log.warn('ARIMA {} failed with ValueError: {}'.format(arima_params, err))
        model_fit = None
    except np.linalg.LinAlgError as err:
        # the "SVD did not converge" can create an error
        log.warn('ARIMA {} failed with LinAlgError: {}'.format(arima_params, err))
        model_fit = None

    return model_fit


class ArimaTrainer(FunctionRegionScalar):
    '''
    A FunctionRegion that uses the train_arima function to train an ARIMA model over a training
    region. Applying this FunctionRegion to the training spatio-temporal region will produce an
    instance of ArimaModelRegion (which is also a SpatialRegion). The shape of ArimaTrainer
    will be [x_len, y_len], where the training region has shape [train_len, x_len, y_len].

    The ArimaModelRegion is also a FunctionRegion, and it will contain a trained ARIMA model
    in each point P(i, j), trained with the training data at P(i, j) of the training region. The
    ArimaModelRegion can later be applied to another region to obtain the ForecastRegion
    (spatio-temporal region).

    This implementation assumes that the same ARIMA hyper-parameters (p, d, q) are used for all
    the ARIMA models.

    To create an instance of this class by using a training region, use the from_training_region()
    class method. This will produce a different ARIMA model in each point.
    '''
    def __init__(self, train_arima_np, forecast_len):
        '''
        Initializes an instance of this function region, which is made of partial calls to
        train_arima(). Since the output of those calls is an object (an ARIMA model), the dtype
        needs to be set to object.

        The forecast length needs to be known now, because apply_to will create an instance of
        ArimaModelRegion (a FunctionRegionSeries), and functions that output series (the forecast)
        need to know the output length in advance.
        '''
        super(ArimaTrainer, self).__init__(train_arima_np, dtype=object)
        self.forecast_len = forecast_len

    def apply_to(self, spt_region):
        '''
        Decorate the default behavior of FunctionRegionScalar.

        The ouput of the parent behavior is to create a SpatialRegion, but we want to create an
        ArimaModelRegion instance. This method will build on the previous result, which already
        has a trained model in each point.
        '''
        # get result from parent behavior
        spatial_region = super(ArimaTrainer, self).apply_to(spt_region)

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

        # return ArimaModelRegion instead of SpatialRegion, in order to create forecasts
        # Since ArimaModelRegion is a FunctionRegionSeries, it requires the length of its output
        # series, that is forecast_len.

        # TODO: should ArimaModelRegion be a SpatialCluster?
        return forecast.ArimaModelRegion(spatial_region.as_numpy, output_len=self.forecast_len)

    @classmethod
    def from_training_region(cls, training_region, arima_params, forecast_len):
        '''
        Creates an instance of this class. This will produce a different ARIMA model in each point.

        training_region
            spatio-temporal region used as training dataset
        arima_params
            ArimaParams hyper-parameters
        '''

        # output shape is given by training shape
        (_, x_len, y_len) = training_region.shape

        # the function signature to be handled by regions can only receive a series
        # so we need to use partial here
        arima_with_params = functools.partial(train_arima, arima_params)

        # the function is applied over all the region, to train models with the same
        # hyperparameters over the training region
        train_arima_np = arrays_util.copy_value_as_matrix_elements(arima_with_params, x_len, y_len)

        return ArimaTrainer(train_arima_np, forecast_len)
