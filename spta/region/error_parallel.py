'''
Perform an error/verification operation in parallel.
This is used to provide a parallel implementation of OverallErrorForEachForecast.

This parallel implementation will iterate over a forecast region. For each point over the
forecast region, allow full access to provided training and observation regions, and perform
an operation.

The operation is assumed to return a scalar value, which is stored in a 2d numpy
array. Parallelism is implemented using multiprocessing.Pool() and 1-d shared arrays. The arrays
are wrapped using a special wrapper, so that the function that iterates over the forecast region
only sees the 3-d objects.

Based on https://research.wmz.ninja/articles/2018/03/
on-sharing-large-arrays-when-using-pythons-multiprocessing.html
'''

import ctypes
import itertools
import numpy as np
import multiprocessing as mp
import os
import sys

# will be shared among processes
global_var_dict = {}


def init_process(forecast_region, observation_region, training_region, output_1d_shared):
    '''
    Initializes the parallel processes with the forecast, training and observation regions.
    Also setups the output array, which is maintained as 1-d.
    '''

    # read-only inputs, will be sent as pickled objects to the processes
    global_var_dict['forecast_region'] = forecast_region
    global_var_dict['observation_region'] = observation_region
    global_var_dict['training_region'] = training_region

    # writeable output, 1-d aray
    global_var_dict['output_1d_shared'] = output_1d_shared


def error_function_wrapper(points_with_error_function):

    # unpack the parameter
    (point, wrapped_error_function) = points_with_error_function

    # Let each process output logs to a separate file, in order to avoid concurrency
    # https://stackoverflow.com/a/23937468
    sys.stdout = open(str(os.getpid()) + ".out", "a")

    # recover the read-only regions
    forecast_region = global_var_dict['forecast_region']
    observation_region = global_var_dict['observation_region']
    training_region = global_var_dict['training_region']

    # data at the specified point
    forecast_series = forecast_region.series_at(point)

    # call wrapped function to perform operation
    error_at_point = wrapped_error_function(point, forecast_series, observation_region,
                                            training_region)

    # prepare 2d representation of the output
    (_, x_len, y_len) = forecast_region.shape
    output_2d = None
    output_2d_shape = (x_len, y_len)

    # need to lock the writeable shared array
    # https://stackoverflow.com/a/7908612
    with global_var_dict['output_1d_shared'].get_lock():

        # recover 2d representation of the output
        output_2d = np.frombuffer(global_var_dict['output_1d_shared'].get_obj())
        output_2d = output_2d.reshape(output_2d_shape)

        # save the operation result
        output_2d[point.x, point.y] = error_at_point

    # no output from parallel function, we are writing to array


class ParallelForecastError(object):
    '''
    A parallel implementation of a generic forecast error function.
    It receives the forecast, observation and training regions. The output is assumed to be
    a spatial region, where each point holds the output of some provided error function evaluated
    with the forecast series of that point:

        output = some SpatialRegion
        output@point = error_function(forecast@point, obs_region, training_region)
    '''

    def __init__(self, num_proc, forecast_region, observation_region, training_region):
        self.num_proc = num_proc
        self.forecast_region = forecast_region
        self.observation_region = observation_region
        self.training_region = training_region

        # initialize the 1-d output array
        # the parallel output is implemented as a 1-d shared array (thread-safe with locking)
        # this needs to be length of (x_len * y_len)
        (_, x_len, y_len) = self.forecast_region.shape
        self.output_1d_len = x_len * y_len
        self.output_2d_shape = (x_len, y_len)
        self.output_1d_shared = mp.Array(ctypes.c_double, self.output_1d_len)

    def operate(self, error_function):
        '''
        Execute the error_function in parallel.
        The signature of error_function must be as follows:
        error_function(point, forecast_series, observation_region, training_region)
        '''

        # initialize the pool, the 'init_process' function will be called with supplied args
        with mp.Pool(processes=self.num_proc, initializer=init_process,
                     initargs=(self.forecast_region, self.observation_region, self.training_region,
                               self.output_1d_shared)) as pool:

            # iterate over the forecast region now to get all points
            # this should work nicely for subclasses of SpatialRegion, e.g. clusters
            points = [point for (point, _) in self.forecast_region]

            # we need to pass a single iterable to imap, where each item is a tuple of the
            # point and the wrapped error function
            # this idiom achieves the effect
            points_with_error_function = zip(points, itertools.repeat(error_function))

            # put the processes to work using map
            # imap doesn't work for some reason, outputs error = 0.0 and no .out files
            # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
            pool.map(error_function_wrapper, points_with_error_function, chunksize=10)

        # the parallel processes are finished now

        # prepare a 2-d view of the output
        # since the output is thread-safe, we need the lock again
        output_2d = None
        with self.output_1d_shared.get_lock():
            output_2d = np.frombuffer(self.output_1d_shared.get_obj())
            output_2d = output_2d.reshape(self.output_2d_shape)

        # ask the forecast region for a 2-d region with the same shape (and subclass)
        decorated_region = self.forecast_region.empty_region_2d()

        # we need to write inside the array
        # trying to assign output_2d to the decorated region does not work,
        # because it is shared memory
        np.copyto(decorated_region.as_numpy, output_2d)

        # to avoid circular imports
        from spta.region.error import ErrorRegion
        return ErrorRegion(decorated_region)
