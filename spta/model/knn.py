'''
Implementation of ModelRegion for the K-Neareast neighbors algorithm (k-NN).
'''

from collections import namedtuple
import numpy as np

from .base import ModelRegion
from .train import ModelTrainer

from spta.util import arrays as arrays_util
from spta.util import log as log_util


class KNNParams(namedtuple('KNNParams', 'k distance_measure')):
    '''
    Parameters for KNN model
    '''
    __slots__ = ()

    def __repr__(self):
        '''
        Override the representation of KNNParams
        # https://stackoverflow.com/a/7914212/3175179
        '''

        # assuming DTW, could be RMSE or some other euclidean distance
        as_str = 'knn-k{}-dtw'
        return as_str.format(self.k)

    def __str__(self):
        return super(KNNParams, self).__repr__()


class KNNModel(namedtuple('KNNModel', 'model_params, training_series')):
    '''
    A model for k-NN. Since we cannot do any useful computation during training (need the forecast length),
    we store the input to compute the forecast during the prediction.

    This is important because the solver expects a proper object (type=object) to be stored as the trained
    model, the object will be an instance of this enum class.
    '''

    __slots__ = ()

    def __repr__(self):
        '''
        Override the representation of KNNModel
        # https://stackoverflow.com/a/7914212/3175179
        '''
        return repr(self.model_params)

    def __str__(self):
        return 'KNNModel(model_params={}, training_series=<{}>)'.format(self.model_params, len(self.training_series))


class ModelRegionKNN(ModelRegion):
    '''
    A FunctionRegion that creates a forecast region obtained by applying the k-NN algorithm for
    a time-series, with size "forecast_len".
    '''

    def forecast_from_model(self, model_at_point, forecast_len, value_at_point, point):
        # see TrainerKNN for the k-NN model
        (model_params, training_series_at_point) = model_at_point.model_params, model_at_point.training_series
        return predict_future_values_with_knn(time_series=training_series_at_point,
                                              k=model_params.k,
                                              forecast_len=forecast_len,
                                              distance_measure=model_params.distance_measure)

    def instance(self, model_numpy_array):
        return ModelRegionKNN(model_numpy_array)


class TrainerKNN(ModelTrainer):
    '''
    A function region used to create instances of TrainerKNN.
    With the current implementation of k-NN for time series, we cannot perform any calculations
    before knowing the value of forecast_len.

    So there is nothing to do here except save the inputs for the k-NN algorithm, that would be
    the k-NN parameters (model_params) and the training series. These are packed in a tuple and
    returned as the "model".
    '''

    def __init__(self, model_params, x_len, y_len):
        super(TrainerKNN, self).__init__(model_params_and_shape=(model_params, x_len, y_len))

    def training_function(self, model_params, training_series):
        '''
        There is nothing to do here except save the inputs for the k-NN algorithm, that would be
        the k-NN parameters (model_params) and the training series. These are packed in an instance
        of KNNModel and returned as the "model".
        '''
        return KNNModel(model_params, training_series)

    def create_model_region(self, numpy_model_array):
        return ModelRegionKNN(numpy_model_array)


def predict_future_values_with_knn(time_series, k, forecast_len, distance_measure):
    '''
    An implementation of k-NN algorithm for predicting the forecast_len next values of a time series.
    The procedure is as follows:

    1. Use the time_series (a 1-d array) to create sliding windows of size forecast_len.
    2. Look at the last window of size forecast_len, and find its k nearest neighbors using the provided
       distance measure.
    3. For each index in the output (size forecast_len), calculate the mean of the k values of the nearest
       neighbors at the corresponding index: y[j] = (1/forecast_len) sum(nn_i [j], i=0, i<k)
    '''
    logger = log_util.logger_for_me(predict_future_values_with_knn)

    # create the sliding windows (neighbors)
    windows = arrays_util.sliding_window(time_series, forecast_len, stride=1)
    (last_window, possible_neighbors) = (windows[-1], windows[:-1])

    # find the k nearest neighbors of the last window
    k_neighbor_indices = find_k_nearest_neighbors(last_window, k, possible_neighbors, distance_measure)
    logger.debug('found indices {}'.format(k_neighbor_indices))
    k_neighbors = possible_neighbors.take(k_neighbor_indices, axis=0)
    logger.debug('k nearest neighbors {}'.format(k_neighbors))

    # the prediction is the mean of the k-NN series
    return np.mean(k_neighbors, axis=0)


def find_k_nearest_neighbors(array, k, possible_neighbors, distance_measure):
    '''
    Given an array of some length 'len' and a tuple of arrays, also of the same length 'len',
    find the k nearest neighbors of the first array among the tuple of arrays, using the provided
    distance measure.

    Returns a tuple of the indices of the nearest neighbors.
    '''

    # sanity check: all neighbors need to have the same length as the array
    equal_len = [len(neighbor) != len(array) for neighbor in possible_neighbors]
    if np.any(equal_len):
        raise ValueError('Neighbors do not have required length of {}!'.format(len(array)))

    # sanity check: not enough neighbors
    if len(possible_neighbors) < k:
        raise ValueError('Not enough neighbors! k={}, neighbors={}'.format(k, len(possible_neighbors)))

    # boundary case: there are k possible neighbors, return all indices
    if len(possible_neighbors) == k:
        return tuple(range(0, k))

    # calculate distances from array to each neighbor
    distances_to_possible_neighbors = [
        distance_measure.measure(array, possible_neighbor)
        for possible_neighbor in possible_neighbors
    ]

    # find the indices of the k-lowest distances
    indices_k_lowest_distances = np.argpartition(np.array(distances_to_possible_neighbors), k)[:k]
    return tuple(indices_k_lowest_distances)
