'''
Unit tests for spta.region.function module.
'''

import numpy as np
import unittest

from spta.region import Point
from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.region.function import FunctionRegionScalar, FunctionRegionScalarSame, \
    FunctionRegionSeries, FunctionRegionSeriesSame

from spta.tests.stub import stub_region


class TestFunctionRegionScalar(unittest.TestCase):
    '''
    Unit tests for function.FunctionRegionScalar
    '''

    def test_square_function(self):

        # create a spatial region
        shape = (3, 5)
        data2d = np.arange(15).reshape(shape)
        sp_region = SpatialRegion(data2d)

        # create a function region that computes the square of each point value
        def square(x):
            return x**2

        square_function_list = [
            square
            for i in range(0, 15)
        ]
        square_function_np = np.array(square_function_list).reshape(shape)
        square_function_region = FunctionRegionScalar(square_function_np)

        # apply the function
        output_region = square_function_region.apply_to(sp_region)

        # test some values
        self.assertEqual(output_region.value_at(Point(0, 0)), 0)
        self.assertEqual(output_region.value_at(Point(0, 2)), 4)
        self.assertEqual(output_region.value_at(Point(2, 4)), 14 * 14)

    def test_mean_function(self):

        # create a spatio temporal region
        # region of 2x3, and a series of length 5 in each point
        shape = (5, 2, 3)
        sptr_data = np.zeros(shape)

        # fill some values
        sptr_data[:, 0, 0] = np.array([1, 2, 3, 4, 5])  # mean 3
        sptr_data[:, 0, 1] = np.array([3, 3, 4, 5, 5])  # mean 4
        sptr_data[:, 0, 2] = np.array([1, 1, 1, 1, 1])  # mean 1
        sptr_data[:, 1, 0] = np.array([5, 6, 7, 8, 9])  # mean 7
        sptr_data[:, 1, 1] = np.array([11, 9, 8, 7, 5])  # mean 8
        sptr_data[:, 1, 2] = np.array([0, 0, 0, 0, 0])  # mean 0

        spt_region = SpatioTemporalRegion(sptr_data)

        # # create the function region that calculates the mean of each series
        # mean_function_list = [
        #     np.mean
        #     for i in range(0, 6)
        # ]
        # mean_function_np = np.array(mean_function_list).reshape((2, 3))
        # mean_function_region = FunctionRegionScalar(mean_function_np)
        mean_function_region = stub_region.stub_mean_function_scalar()

        # apply the function
        output_region = mean_function_region.apply_to(spt_region)

        # test values
        self.assertEqual(output_region.value_at(Point(0, 0)), 3)
        self.assertEqual(output_region.value_at(Point(0, 1)), 4)
        self.assertEqual(output_region.value_at(Point(0, 2)), 1)
        self.assertEqual(output_region.value_at(Point(1, 0)), 7)
        self.assertEqual(output_region.value_at(Point(1, 1)), 8)
        self.assertEqual(output_region.value_at(Point(1, 2)), 0)


class TestFunctionRegionScalarSame(unittest.TestCase):
    '''
    Unit tests for function.FunctionRegionScalarSame
    '''

    def test_square(self):

        # create a spatial region
        shape = (3, 5)
        data2d = np.arange(15).reshape(shape)
        sp_region = SpatialRegion(data2d)

        # create a function region that computes the square of each point value
        def square(x):
            return x**2

        square_function_region = FunctionRegionScalarSame(square, shape[0], shape[1])

        # apply the function
        output_region = square_function_region.apply_to(sp_region)

        # test some values
        self.assertEqual(output_region.value_at(Point(0, 0)), 0)
        self.assertEqual(output_region.value_at(Point(0, 2)), 4)
        self.assertEqual(output_region.value_at(Point(2, 4)), 14 * 14)


class TestFunctionRegionSeries(unittest.TestCase):
    '''
    Unit tests for function.FunctionRegionSeries
    '''

    def test_sum_and_mean_function(self):

        # create a spatio temporal region
        # region of 2x3, and a series of length 5 in each point
        shape = (5, 2, 3)
        sptr_data = np.zeros(shape)

        # fill some values
        sptr_data[:, 0, 0] = np.array([1, 2, 3, 4, 5])  # mean 3
        sptr_data[:, 0, 1] = np.array([3, 3, 4, 5, 5])  # mean 4
        sptr_data[:, 0, 2] = np.array([1, 1, 1, 1, 1])  # mean 1
        sptr_data[:, 1, 0] = np.array([5, 6, 7, 8, 9])  # mean 7
        sptr_data[:, 1, 1] = np.array([11, 9, 8, 7, 5])  # mean 8
        sptr_data[:, 1, 2] = np.array([0, 0, 0, 0, 0])  # mean 0

        spt_region = SpatioTemporalRegion(sptr_data)

        # create a function region that computes the sum and mean of a series
        def sum_and_mean(series):
            result = np.array((np.sum(series), np.mean(series)))
            return result

        sum_and_mean_list = [
            sum_and_mean
            for i in range(0, 6)
        ]

        sum_and_mean_np = np.array(sum_and_mean_list).reshape((2, 3))
        sum_and_mean_region = FunctionRegionSeries(sum_and_mean_np)

        # apply the function
        output_region = sum_and_mean_region.apply_to(spt_region, output_len=2)

        # test values
        result_0_0 = output_region.series_at(Point(0, 0))
        self.assertEqual(result_0_0.tolist(), [15, 3])
        result_0_1 = output_region.series_at(Point(0, 1))
        self.assertEqual(result_0_1.tolist(), [20, 4])
        result_0_2 = output_region.series_at(Point(0, 2))
        self.assertEqual(result_0_2.tolist(), [5, 1])
        result_1_0 = output_region.series_at(Point(1, 0))
        self.assertEqual(result_1_0.tolist(), [35, 7])
        result_1_1 = output_region.series_at(Point(1, 1))
        self.assertEqual(result_1_1.tolist(), [40, 8])
        result_1_2 = output_region.series_at(Point(1, 2))
        self.assertEqual(result_1_2.tolist(), [0, 0])


class TestFunctionRegionSeriesSame(unittest.TestCase):
    '''
    Unit tests for function.FunctionRegionSeriesSame
    '''

    def test_sum_and_mean_function(self):

        # create a spatio temporal region
        # region of 2x3, and a series of length 5 in each point
        shape = (5, 2, 3)
        sptr_data = np.zeros(shape)

        # fill some values
        sptr_data[:, 0, 0] = np.array([1, 2, 3, 4, 5])  # mean 3
        sptr_data[:, 0, 1] = np.array([3, 3, 4, 5, 5])  # mean 4
        sptr_data[:, 0, 2] = np.array([1, 1, 1, 1, 1])  # mean 1
        sptr_data[:, 1, 0] = np.array([5, 6, 7, 8, 9])  # mean 7
        sptr_data[:, 1, 1] = np.array([11, 9, 8, 7, 5])  # mean 8
        sptr_data[:, 1, 2] = np.array([0, 0, 0, 0, 0])  # mean 0

        spt_region = SpatioTemporalRegion(sptr_data)

        # create a function region that computes the sum and mean of a series
        def sum_and_mean(series):
            result = np.array((np.sum(series), np.mean(series)))
            return result

        sum_and_mean_region = FunctionRegionSeriesSame(sum_and_mean, shape[1], shape[2])

        # apply the function
        output_region = sum_and_mean_region.apply_to(spt_region, output_len=2)

        # test values
        result_0_0 = output_region.series_at(Point(0, 0))
        self.assertEqual(result_0_0.tolist(), [15, 3])
        result_0_1 = output_region.series_at(Point(0, 1))
        self.assertEqual(result_0_1.tolist(), [20, 4])
        result_0_2 = output_region.series_at(Point(0, 2))
        self.assertEqual(result_0_2.tolist(), [5, 1])
        result_1_0 = output_region.series_at(Point(1, 0))
        self.assertEqual(result_1_0.tolist(), [35, 7])
        result_1_1 = output_region.series_at(Point(1, 1))
        self.assertEqual(result_1_1.tolist(), [40, 8])
        result_1_2 = output_region.series_at(Point(1, 2))
        self.assertEqual(result_1_2.tolist(), [0, 0])
