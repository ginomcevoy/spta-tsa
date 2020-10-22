import functools
import numpy as np

from spta.region import TimeInterval
from spta.region.function import FunctionRegionScalar

from spta.util import log as log_util


class ModelTrainer(FunctionRegionScalar):
    '''
    A FunctionRegion which is meant to be applied to a training spatio-temporal region, in order to
    produce an instance of a ModelRegion subclass.

    Subclasses must provide the implementations of the methods train_function_by_point() and
    create_model_region().

    The method training_function() is used to provide the function that will be applied to the
    training region at each point. This completes the construction of the internal numpy_dataset that
    contains the function objects at each point. The FunctionRegion interface assumes that the only
    parameter of the function is the value of the region at each point, so functools.partial is used
    to pass the model parameters to the function and maintain the interface.
    '''

    def __init__(self, model_params_and_shape=None, model_params_region=None):
        '''
        To create an instance of ModelTrainer, either provide model_params_and_shape or model_params_region,
        they are mutually exclusive.

        model_params_and_shape:
            A tuple (model_params, x_len, y_len). If this value is provided, then assume that the region
            will have the shape (x_len, y_len), and that the model parameters are constant
        '''

        # sanity checks
        if model_params_and_shape is None and model_params_region is None:
            raise ValueError('Provide either model_params_and_shape or model_params_region')

        if model_params_and_shape is not None and model_params_region is not None:
            raise ValueError('Cannot provide both model_params_and_shape or model_params_region')

        if model_params_and_shape is not None:

            # use constant parameters: the shape needs to be provided
            (model_params, x_len, y_len) = model_params_and_shape

            numpy_function_array = np.empty((x_len, y_len), dtype=object)
            for x in range(x_len):
                for y in range(y_len):
                    numpy_function_array[x][y] = functools.partial(self.training_function, model_params)

        if model_params_region is not None:

            numpy_function_array = np.empty((model_params_region.x_len, model_params_region.y_len),
                                            dtype=object)

            # use the iterator of the region: if the region is a cluster, a nested for
            # will fail because value_at(point) raises ValueError if point is not a cluster member.
            # if a point (x, y) is not iterated, the corresponding model_params will be None.
            for point, model_params_for_point in model_params_region:
                x, y = (point.x, point.y)
                numpy_function_array[x][y] = functools.partial(self.training_function, model_params_for_point)

        # use dtype=object to indicate that we are storing objects (functions)
        super(ModelTrainer, self).__init__(numpy_function_array, dtype=object)

        # useful for decorators
        self.model_params_and_shape = model_params_and_shape
        self.model_params_region = model_params_region

    def apply_to(self, training_region):
        '''
        Override the parent behavior of FunctionRegionScalar: instead of returning a value for f_{(x,y)}(x,y),
        we must return an instance of the trained model as returned by training_function() abstract method.

        Also, the output of the parent behavior is a SpatialRegion, but we want to create a subclass of
        ModelRegion instead. The parent SpatialRegion contains the trained model at training region,
        so this method is decorated to achieve the effect.
        '''
        # get result from parent behavior
        # this will already call the training function (see constructor!) and return trained models.
        spatial_region_with_models = super(ModelTrainer, self).apply_to(training_region)

        # count and log missing models, iterate to find them
        self.missing_count = 0
        for (point, trained_model) in spatial_region_with_models:

            if trained_model is None:
                self.missing_count += 1

        if self.missing_count:
            self.logger.warn('Missing models: {}'.format(self.missing_count))
        else:
            self.logger.info('Models trained in all points successfully.')

        # decorate the output by returning the desired instance (subclasses define the correct instance)
        return self.create_model_region(spatial_region_with_models.as_numpy)

    def training_function(self, model_params, training_series):
        '''
        Here the training takes place at each point of the region. The model_params and training_series match
        for some point P(x, y).

        The output must be an object representing the trained model which can later be used to create a forecast.
        If the model cannot be trained, then this method must return None.
        '''
        raise NotImplementedError

    def create_model_region(self, numpy_model_array):
        '''
        Return an instance of ModelRegion (not checking here if the instance matches ModelRegion)
        '''
        raise NotImplementedError


class NoTrainer(ModelTrainer):
    '''
    A trainer that does not train anything. Instead, it already has the models.
    '''

    def __init__(self, trained_models):
        super(ModelTrainer, self).__init__(model_params_region=trained_models)
        self.trained_models = trained_models

    def apply_to(self, training_region):
        self.missing_count = 0
        return self.trained_models


class TrainAtRepresentatives(ModelTrainer):
    '''
    A decorator of ModelTrainer that only trains models at the specified points (representatives of the region).
    Other points will get a None model.
    '''

    def __init__(self, decorated, representatives):
        '''
        decorated:
            Instance of a subclass of ModelTrainer.

        representatives:
            a list of points where the models will be computed, e.g. medoids of a cluster partition
        '''
        super(TrainAtRepresentatives, self).__init__(model_params_and_shape=decorated.model_params_and_shape,
                                                     model_params_region=decorated.model_params_region)
        self.decorated = decorated
        self.representatives = representatives

    def function_at(self, point):
        '''
        Here we decorate the behavior: models will be trained only at the representative points,
        other points will return None (no model)
        '''
        if point in self.representatives:
            return self.decorated.function_at(point)

        else:
            def function_that_returns_no_model(training_series):
                return None

            return function_that_returns_no_model

    def apply_to(self, training_region):
        # don't apply as the decorated! we need the modified function_at to work
        return super(TrainAtRepresentatives, self).apply_to(training_region)

    def training_function(self, model_params, training_series):
        # use the decorated training function
        return self.decorated.training_function(model_params, training_series)

    def create_model_region(self, numpy_model_array):
        # create the region as the decorated mandates
        # but before that, we need to recover self.missing count, because this
        # decorator has it but the decorated does not
        self.decorated.missing_count = self.missing_count
        return self.decorated.create_model_region(numpy_model_array)


class SplitTrainingAndTest(log_util.LoggerMixin):

    def __init__(self, test_len):
        super(SplitTrainingAndTest, self).__init__()
        self.test_len = test_len

    def split(self, spt_region):
        raise NotImplementedError


class SplitTrainingAndTestLast(SplitTrainingAndTest):
    '''
    Use the last entries elements in each time series, to split a spatio-temporal region into
    training region and test region.
    '''

    def split(self, spt_region):
        '''
        Given a spatio-temporal region, split the time series in two, to get a training region
        and a test region.

        spt_region
            spatio-temporal region with shape [(training_len + test_len), x_len, y_len]

        test_len
            Size of the test series that will be separated to create a test region. The rest of the
            series will form the training region.

        Output is a tuple of two spatio-temporal regions:
            training subset: shape [training_len, x_len, y_len]
            test_subset: shape [test_len, x_len, y_len]
        '''
        series_len = spt_region.series_len

        # use the concept of time intervals to easily split the region by the temporal axis
        training_size = series_len - self.test_len
        training_interval = TimeInterval(0, training_size)
        test_interval = TimeInterval(training_size, series_len)

        training_subset = spt_region.interval_subset(training_interval)
        test_subset = spt_region.interval_subset(test_interval)

        training_subset.name = 'train_{}'.format(spt_region)
        test_subset.name = 'test_{}'.format(spt_region)

        self.logger.debug('training_subset: {} -> {}'.format(training_subset,
                                                             training_subset.shape))
        self.logger.debug('test_subset: {} -> {}'.format(test_subset, test_subset.shape))

        return (training_subset, test_subset)
