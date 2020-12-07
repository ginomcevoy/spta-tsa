import numpy as np

from . import maths as maths_util


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

    # don't try to unpack a namedtuple object that contains a series...
    if hasattr(value, '_fields'):
        # the value is a namedtuple, we want to copy the whole instance instead of its elements
        # hack based on https://stackoverflow.com/a/53577004
        same_value_mn_times = [value for i in range(0, m * n)]
        same_value_mn_times = np.array(same_value_mn_times + [None], object)[:-1]

    else:
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


def mean_squared(array):
    '''
    Mean Squared calculation, ignores NaN values. This implementation does not let NaN
    values affect the weight of other values.
    '''
    return np.nanmean(np.array(array)**2)


def root_mean_squared(array):
    '''
    Root Mean Squared calculation, just the square root of mean_squared().
    '''
    return (mean_squared(array))**0.5


def list_of_2d_points(x_len, y_len):
    '''
    Given (x_len, y_len), outputs a list of points of the corresponding 2d region:
    (0, 0), (0, 1), ... (0, y_len - 1), (1, 0), (1, 1), ... (x_len - 1, y_len -1)
    '''
    result = np.empty((x_len * y_len, 2), dtype=np.uint32)

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


def regular_partitioning(x_len, y_len, k):
    '''
    Given a 2-d region and a number of clusters k, create a membership matrix that can be used
    to create clusters of (approximately) equal size.

    The process is as follows:
        - Compute the most balanced divisors of k, e.g. 12 -> 4 x 3.
        - For simplicity, validate that x_len >= div_x and y_len >= div_y, the divisors
        - Partition the (x_len, y_len)  using these two values to create the cluster labels, e.g.
                  0  0  0  1  1  1 ...  3  3  3
                  0  0  0  1  1  1 ...  3  3  3
                  .....................
                  9  9  9 10 10 10 ..  11 11 11
                  9  9  9 10 10 10 ..  11 11 11

        - Return this matrix
    '''

    # this finds 12 -> 3x4 or 24 -> 4x6. Use the largest to divide the largest of x_len, y_len
    (div_x, div_y) = maths_util.two_balanced_divisors_order_x_y(k, x_len, y_len)

    # bounds check
    if x_len < div_x or y_len < div_y:
        raise ValueError('Region ({}, {}) too small for k={}!'.format(x_len, y_len, k))

    # find the size of each partition here
    # this division may be inexact
    lines_per_row = int(x_len / div_x)
    lines_per_col = int(y_len / div_y)

    # handle inexact division
    # two scenarios: residue is small ( < lines_per_* / 2) or large ( >= lines_per_* / 2)
    # second scenario is handled below
    residue_row = x_len % div_x
    residue_col = y_len % div_y

    # handle first scenario: add more lines per rows/columns as needed

    # we have an "edge" case if div_* is large, in that case we cannot accommodate the first
    # scenario to use an additional line per row/column
    # this may happen when k is prime and the region is not large enough
    # see test_regular_partitioning_whole_real_brazil_in_23 in test_arrays
    def can_increase_lines_per_cluster(divisor, residue, lines_per_cluster):
        return residue > divisor * (divisor - lines_per_cluster - 1)

    # handle row here
    if residue_row > lines_per_row / 2 and \
            can_increase_lines_per_cluster(div_x, residue_row, lines_per_row):
        lines_per_row += 1

        # recompute the residue as negative!
        # this will be picked up by second scenario and shrink the last row/col
        residue_row = x_len - lines_per_row * div_x

    # same for column
    if residue_col > lines_per_col / 2 and \
            can_increase_lines_per_cluster(div_y, residue_col, lines_per_col):

        lines_per_col += 1
        residue_col = y_len - lines_per_col * div_y

    # prepare output, same shape as input region
    # type is integer for the labels!
    matrix = np.zeros((x_len, y_len), dtype=np.uint32)

    # iterate each cluster to set labels i in [0, k-1]
    # do this by iterating rows and cols of partition
    for row in range(0, div_x):
        for col in range(0, div_y):

            # the label for the current cluster
            i = int(row * div_y + col)

            # the boundaries for the cluster
            # exact for now
            x_start = lines_per_row * row
            y_start = lines_per_col * col

            x_end = x_start + lines_per_row
            y_end = y_start + lines_per_col

            # handle residues here

            # first scenario: small residues
            # the second case degrades to the first case here... (we increased lines_per_col)
            # so no need to check if residue_* < lines_per_* / 2
            # exact case should also fall naturally on this one
            if col == div_y - 1:
                # handle small column-wise residues: add them to last column
                y_end += residue_col

            if row == div_x - 1:
                # handle small row-wise residues: add them to last row
                x_end += residue_row

            # mark labels
            matrix[x_start:x_end, y_start:y_end] = i

    return matrix


def sliding_window(array, window_len, stride=1):
    '''
    Create a sliding window for a given 1-d array with the given stride. The stride is the amount of units
    that the next window is displaced, relative the current one.
    '''

    windows = []
    for i in range(0, len(array) - window_len + 1, stride):
        window = array[i:(i + window_len)]
        windows.append(window)

    return np.array(windows)


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
