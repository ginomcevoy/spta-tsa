import numpy as np


def copy_array_as_matrix_elements(array, m, n):
    '''
    Given an array, create the following mxn matrix:

    [ <array>  <array> ... <array>
      <array>  <array> ... <array>
      ...
      <array>  <array> ... <array> ] (mxn)

    This allows us to create a SpatioTemporalRegion where all the temporal series are
    the same as the array.
    '''
    # if we input a matrix and ask for axis=0, we get the effect of copying the array mxn times
    length = len(array)
    s = [array]
    mat = np.repeat(s, repeats=m * n, axis=0)
    return mat.reshape(m, n, length)


def spatio_temporal_to_list_of_time_series(numpy_dataset):
    (x_len, y_len, series_len) = numpy_dataset.shape
    elements_1d = numpy_dataset.reshape(x_len * y_len * series_len)
    return np.split(elements_1d, x_len * y_len)


def minimum_value_and_index(numpy_dataset):
    '''
    Given a n-dimensional array, finds the overall minimum value.
    Returns the value and the index in the dataset. The index is expressed as a n-tuple
    (n dimensions)
    '''
    # overall min, ignore NaN
    minimum = np.nanmin(numpy_dataset)
    index_as_tuple_of_arrays = np.where(numpy_dataset == minimum)
    index = [
        index_coord[0]
        for index_coord
        in index_as_tuple_of_arrays
    ]
    return (minimum, tuple(index))


def root_sum_squared(array):
    return np.sum(np.array(array)**2)**.5


def root_mean_squared(array):
    '''
    Root Mean Squared calculation, ignores NaN values. This implementation does not let NaN
    values affect the weight of other values.
    '''
    return (np.nanmean(np.array(array)**2))**0.5


if __name__ == '__main__':

    x = (0, 1, 2, 3, 4)
    print('WTF')
    xx = copy_array_as_matrix_elements(x, 3, 4)
    print('xx: %s' % (xx.shape,))
    print(xx)
    print(xx[2, 1])

    xx_2d = spatio_temporal_to_list_of_time_series(xx)
    msg = 'spatio_temporal_to_list_of_time_series: %s lists of array with %s elems'
    print(msg % (len(xx_2d), xx_2d[0].shape,))
    print(xx_2d)

    print('minimum_value_and_index 1d')
    r1d = np.random.rand(5)
    print(r1d)
    (m, i) = minimum_value_and_index(r1d)
    print(m, i)

    print('minimum_value_and_index 2d')
    r2d = np.random.rand(3, 4)
    print(r2d)
    (m, i) = minimum_value_and_index(r2d)
    print(m, i)
