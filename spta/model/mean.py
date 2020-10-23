from collections import namedtuple
import numpy as np

from .base import ModelRegion
from .train import ModelTrainer


class MeanOfPastParams(namedtuple('MeanOfPastParams', 'past')):
    '''
    Parameters for MeanOfPast model
    '''
    __slots__ = ()

    def __repr__(self):
        '''
        Override the representation of MeanOfPastParams
        # https://stackoverflow.com/a/7914212/3175179
        '''
        as_str = 'mean-past{}'
        return as_str.format(self.past)

    def __str__(self):
        return super(MeanOfPastParams, self).__repr__()


class ModelRegionMeanOfPast(ModelRegion):
    '''
    A FunctionRegion that creates a forecast region using a constant mean value of
    the training series
    '''

    def forecast_from_model(self, model_at_point, forecast_len, value_at_point, point):
        # assume that the "model" is actually the mean of the training series
        # repeat it forecast_len times
        return np.repeat(model_at_point, forecast_len)

    def instance(self, model_numpy_array):
        return ModelRegionMeanOfPast(model_numpy_array)


class TrainerMeanOfPast(ModelTrainer):
    '''
    A function region used to create instances of ModelRegionMeanOfPast.
    Uses the mean of the last 'past' values of the training region, where 'past'
    is the only parameter of the model.
    '''

    def __init__(self, model_params, x_len, y_len):
        super(TrainerMeanOfPast, self).__init__(model_params_and_shape=(model_params, x_len, y_len))

    def training_function(self, model_params, training_series):
        '''
        Create the 'model' for a single point: calculate the mean of the training series,
        then store this mean
        '''

        # sanity check: no data means no model
        # useful for sparse datasets or when refitting
        if training_series is None:
            return None

        self.logger.debug('Training MeanOfPast with training size {}'.format(len(training_series)))

        past_values = training_series[-model_params.past:]
        return np.mean(past_values)

    def create_model_region(self, numpy_model_array):
        return ModelRegionMeanOfPast(numpy_model_array)
