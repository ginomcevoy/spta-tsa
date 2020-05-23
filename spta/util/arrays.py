import numpy as np


def copy_array_as_matrix_elements(array, m, n):
    '''
    Given an array with length l, create a lxmxn matrix such that matrix[:, x, y] = array for all
    x, y in range.

    This allows us to create a SpatioTemporalRegion where all the temporal series are
    the same as the array.
    '''
    length = len(array)
    # s = [array]
    mat = np.repeat(array, repeats=m * n)  # , axis=0)
    return mat.reshape(length, m, n)


def copy_value_as_matrix_elements(value, m, n):
    '''
    Given a value, create a mxn matrix such that matrix[i, j] = value for all i, j in range.

    This allows us to create a SpatialRegion where all the values are the same as provided.
    '''
    same_value_mn_times = np.repeat(value, repeats=(m * n))
    return same_value_mn_times.reshape(m, n)


def spatio_temporal_to_list_of_time_series(numpy_dataset):
    (series_len, x_len, y_len) = numpy_dataset.shape
    # elements_1d = numpy_dataset.reshape(series_len * x_len * y_len)
    # return np.split(elements_1d, x_len * y_len
    list_of_time_series = []
    for x in range(0, x_len):
        for y in range(0, y_len):
            list_of_time_series.append(numpy_dataset[:, x, y])

    return list_of_time_series


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


def maximum_value_and_index(numpy_dataset):
    '''
    Given a n-dimensional array, finds the overall maximum value.
    Returns the value and the index in the dataset. The index is expressed as a n-tuple
    (n dimensions)
    '''
    # overall max, ignore NaN
    maximum = np.nanmax(numpy_dataset)
    index_as_tuple_of_arrays = np.where(numpy_dataset == maximum)
    index = [
        index_coord[0]
        for index_coord
        in index_as_tuple_of_arrays
    ]
    return (maximum, tuple(index))


def root_sum_squared(array):
    return np.sum(np.array(array)**2)**.5


def root_mean_squared(array):
    '''
    Root Mean Squared calculation, ignores NaN values. This implementation does not let NaN
    values affect the weight of other values.
    '''
    return (np.nanmean(np.array(array)**2))**0.5


def list_of_2d_points(x_len, y_len):
    '''
    Given (x_len, y_len), outputs a list of points of the corresponding 2d region:
    (0, 0), (0, 1), ... (0, y_len - 1), (1, 0), (1, 1), ... (x_len - 1, y_len -1)
    '''
    result = np.empty((x_len * y_len, 2), dtype=np.int8)

    for i in range(0, x_len):
        for j in range(0, y_len):
            index = i * y_len + j
            result[index, :] = (i, j)

    return result


def distances_to_point_in_2d_region(i, j, x_len, y_len):
    '''
    Given a 2d region (x_len, y_len) and a point (i, j), returns a list (x_len * y_len, 1) of
    euclidian distances from each point to the (i,j) point.
    '''
    single_point = [i, j]
    points_of_2d_region = list_of_2d_points(x_len, y_len)
    return np.linalg.norm(points_of_2d_region - single_point, axis=1)


if __name__ == '__main__':

    x = (0, 1, 2, 3, 4)
    xx = copy_array_as_matrix_elements(x, 3, 4)
    print('xx: %s' % (xx.shape,))
    print(xx)
    print(xx[:, 2, 1])

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

    print('list_of_2d_points(4, 5)')
    r = list_of_2d_points(4, 5)
    print(r)

    print('distances_to_point_in_2d_region(2, 3, 4, 5)')
    r = distances_to_point_in_2d_region(2, 3, 4, 5)
    print(r)

    print('last distances, reshaped:')
    print(r.reshape(4, 5))
