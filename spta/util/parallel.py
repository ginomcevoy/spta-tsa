# Based on
# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

import ctypes
import numpy as np
import multiprocessing as mp
import itertools
import os
import sys

from spta.region.temporal import SpatioTemporalRegion

# will be shared among processes
global_var_dict = {}


def init_process(sptr_1d_shared, output_1d_shared, series_len_shared, x_len_shared, y_len_shared):
    '''
    Initializes the processes with the shared input, shared output and the 2D shape.
    '''
    global_var_dict['sptr_1d_shared'] = sptr_1d_shared
    global_var_dict['output_1d_shared'] = output_1d_shared
    global_var_dict['series_len_shared'] = series_len_shared
    global_var_dict['x_len_shared'] = x_len_shared
    global_var_dict['y_len_shared'] = y_len_shared


def inter_points_task_wrapper(k_with_task):
    '''
    Convenient wrapper to hide the underlying 1D implementation.
    Lets external function work with two indices for inter-point operations.
    '''
    (k_index, inter_points_task) = k_with_task

    # read shape from shared values
    series_len = global_var_dict['series_len_shared'].value
    x_len = global_var_dict['x_len_shared'].value
    y_len = global_var_dict['y_len_shared'].value

    # recover the indices of the two points
    # the first x_len * y_len k_indices are for operations between the first point and all the
    # other points.
    i = int(k_index / (x_len * y_len))
    j = int(k_index % (x_len * y_len))

    # redirect stdout to a file for parallel 'logging'
    # https://stackoverflow.com/a/23937468
    sys.stdout = open(str(os.getpid()) + ".out", "a")
    # print('Got k={}'.format(k_index), flush=True)
    # print('({}, {})'.format(i, j), flush=True)

    # recover 3D representation of spatio-temporal region
    spt_shape = (series_len, x_len, y_len)
    spt_region_np = np.frombuffer(global_var_dict['sptr_1d_shared']).reshape(spt_shape)
    spt_region = SpatioTemporalRegion(spt_region_np)

    # call wrapped function to perform operation
    result_i_j = inter_points_task(spt_region, i, j)

    # recover the 2d representation of the output
    # need to lock the writeable array
    output_2d = None
    output_2d_shape = (x_len * y_len, x_len * y_len)

    with global_var_dict['output_1d_shared'].get_lock():
        output_2d = np.frombuffer(global_var_dict['output_1d_shared'].get_obj())
        output_2d = output_2d.reshape(output_2d_shape)

        # save the operation result
        output_2d[i, j] = result_i_j

    # we don't need this
    return 0


class InterPointsOperation(object):
    '''
    Given a spatiotemporal region, executes an operation between each two points in the region,
    using parallel workers. Assumes that operations between different points are all independent.

    If the spatio-temporal region has shape (series_len, x_len, y_len), then the output is the
    matrix (x_len * y_len, x_len * y_len).
    Useful for parallel calculation of the distance matrix with DTW.
    '''

    def __init__(self, num_proc, spt_region):
        self.num_proc = num_proc

        (series_len, x_len, y_len) = self.__init_input(spt_region)
        self.__init_output(x_len, y_len)
        self.__init_shape(series_len, x_len, y_len)

    def __init_input(self, spt_region):
        '''
        Prepare the spatio temporal data to be shared among the workers as a shared,
        read-only array.
        '''
        (series_len, x_len, y_len) = spt_region.shape

        # need to work with a 1D array
        # will be read-only so no locking needed
        data_len = series_len * x_len * y_len
        self.sptr_1d_shared = mp.RawArray('d', data_len)

        # copy the data in the spatio temporal region to shared array, using numpy representation
        sptr_1d_shared_np = np.frombuffer(self.sptr_1d_shared)
        sptr_1d_shared_np = sptr_1d_shared_np.reshape((series_len, x_len, y_len))
        np.copyto(sptr_1d_shared_np, spt_region.as_numpy)

        return (series_len, x_len, y_len)

    def __init_output(self, x_len, y_len):

        # the output matrix will be implemented as a 1D  shared array (thread-safe with locking)
        # this needs to be length of (x_len * y_len)^2 to store all operations between points
        self.output_2d_shape = (x_len * y_len, x_len * y_len)
        self.all_ops_len = x_len * y_len * x_len * y_len
        self.output_1d_shared = mp.Array(ctypes.c_double, self.all_ops_len)

        # wrap the inner implementation as a numpy 2D matrix for easier manipulation, initialize
        # need to sync access to shared object
        # https://stackoverflow.com/a/7908612
        with self.output_1d_shared.get_lock():
            self.output_2d = np.frombuffer(self.output_1d_shared.get_obj())
            self.output_2d = self.output_2d.reshape(self.output_2d_shape)
            self.output_2d.fill(0)

    def __init_shape(self, series_len, x_len, y_len):

        # save the region shape as RawValue (no locking), so that all processes can read shape
        self.series_len_shared = mp.RawValue('i', series_len)
        self.x_len_shared = mp.RawValue('i', x_len)
        self.y_len_shared = mp.RawValue('i', y_len)

    def operate(self, inter_points_task):
        '''
        Execute the inter_points_task in parallel.
        The signature of inter_points_task must be as follows:
        inter_points_task(spt_region, i, j)
        '''
        with mp.Pool(processes=self.num_proc, initializer=init_process,
                     initargs=(self.sptr_1d_shared, self.output_1d_shared, self.series_len_shared,
                               self.x_len_shared, self.y_len_shared)) as pool:

            # the size of the queue is (x_len * y_len)^2 to account for all inter-point ops
            k_iter = range(0, self.all_ops_len)

            # we want to pass both k and the wrapped task, but we must pass an interable to
            # pool.map. This achieves the effect.
            k_with_task = zip(k_iter, itertools.repeat(inter_points_task))

            # put the processes to work
            # no need to store result of map, since we are writing to shared array
            pool.map(inter_points_task_wrapper, k_with_task)

        # return 2D representation of the shared output array
        return self.output_2d
