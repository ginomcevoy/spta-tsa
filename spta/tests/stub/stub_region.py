import numpy as np

from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.region.function import FunctionRegionScalar, FunctionRegionSeries


def numpy_2d_stub():
    np_2d = np.arange(20).reshape(4, 5)
    np_2d = np_2d ** 2
    return np_2d


def spatial_region_stub():
    return SpatialRegion(numpy_2d_stub())


def numpy_3d_stub():
    # manual dataset
    series_0_0 = np.array((1, 2, 3))
    series_0_1 = np.array((4, 5, 6))
    series_0_2 = np.array((7, 8, 9))
    series_1_0 = np.array((11, 12, 13))
    series_1_1 = np.array((14, 15, 16))
    series_1_2 = np.array((17, 18, 19))

    # shape: x_len = 2, y_len = 3
    nd = np.empty((3, 2, 3))
    nd[:, 0, 0] = series_0_0
    nd[:, 0, 1] = series_0_1
    nd[:, 0, 2] = series_0_2
    nd[:, 1, 0] = series_1_0
    nd[:, 1, 1] = series_1_1
    nd[:, 1, 2] = series_1_2

    return nd


def numpy_3d_4spd_stub():
    # manual dataset
    series_0_0 = np.array((1, 2, 4, 5))
    series_0_1 = np.array((6, 7, 9, 10))
    series_0_2 = np.array((11, 12, 14, 15))
    series_1_0 = np.array((16, 17, 19, 20))
    series_1_1 = np.array((21, 22, 24, 25))
    series_1_2 = np.array((26, 27, 29, 30))

    # shape: x_len = 2, y_len = 3
    nd = np.empty((4, 2, 3))
    nd[:, 0, 0] = series_0_0
    nd[:, 0, 1] = series_0_1
    nd[:, 0, 2] = series_0_2
    nd[:, 1, 0] = series_1_0
    nd[:, 1, 1] = series_1_1
    nd[:, 1, 2] = series_1_2

    return nd


def spatio_temporal_region_stub():
    numpy_dataset = numpy_3d_stub()
    return SpatioTemporalRegion(numpy_dataset)


def spt_region_stub_2_4_5():
    np_3d = np.arange(40).reshape(2, 4, 5)
    return SpatioTemporalRegion(np_3d)


def stub_mean_function_scalar():
    '''
    A FunctionRegionScalar that calculates the mean of each series
    '''
    mean_function_list = [
        np.mean
        for i in range(0, 6)
    ]
    mean_function_np = np.array(mean_function_list).reshape((2, 3))
    return FunctionRegionScalar(mean_function_np)


def stub_reverse_function_series():
    '''
    A FunctionRegionSeries that reverses the order of each series (creates a new series array)
    Works on 2x3 regions.
    '''
    def reverse_function(series):
        rev_view = series[::-1]
        return np.copy(rev_view)

    reverse_function_list = [
        reverse_function
        for i in range(0, 6)
    ]
    reverse_function_np = np.array(reverse_function_list).reshape((2, 3))
    return FunctionRegionSeries(reverse_function_np)


def verify_result_is_expected(test, result, expected):
    (result_t, result_x, result_y) = result.shape
    (expected_t, expected_x, expected_y) = expected.shape

    test.assertEqual(result_t, expected_t)
    test.assertEqual(result_x, expected_x)
    test.assertEqual(result_y, expected_y)
    # test.fail('{} {}'.format(result_t, expected_t))

    for x in range(0, result_x):
        for y in range(0, result_y):
            result_x_y = result[:, x, y]
            expected_x_y = expected[:, x, y]

            # this idiom compares numpy arrays
            test.assertIsNone(np.testing.assert_array_equal(result_x_y, expected_x_y))
