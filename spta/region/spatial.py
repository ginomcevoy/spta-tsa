import numpy as np

from spta.util import log as log_util

from . import Point, Region, reshape_1d_to_2d


# model + test + forecast -> error

# arimitas -> error de cada arima en su entrenamiento
# 1 model + test/region + forecast/region -> error en cada punto

class SpatialRegion(log_util.LoggerMixin):

    def __init__(self, numpy_dataset, region_metadata=None):
        self.numpy_dataset = numpy_dataset
        self.metadata = region_metadata

        # used for iterating over points
        self.point_index = 0

        # to make iterations faster
        self.shape = numpy_dataset.shape
        self.ndim = numpy_dataset.ndim

        # makes iterations faster
        # this implemetation can be reused for spatio temporal regions with (series, x, y) shape
        self.x_len = self.shape[self.ndim - 2]
        self.y_len = self.shape[self.ndim - 1]

    def __iter__(self):
        '''
        Used for iterating over points
        '''
        return self

    def __next__(self):
        '''
        Used for iterating over points.
        The iterator returns the tuple (Point, value) for each point.
        '''
        # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
        if self.point_index >= self.y_len * self.x_len:
            # stop iteration, but allow reuse of iterator from start again
            self.point_index = 0
            raise StopIteration

        # iterate
        i = int(self.point_index / self.y_len)
        j = self.point_index % self.y_len
        point_i_j = Point(i, j)
        self.point_index += 1

        # tuple output
        return (point_i_j, self.value_at(point_i_j))

    @property
    def as_numpy(self):
        return self.numpy_dataset

    @property
    def as_array(self):
        '''
        Goes from 2d to 1d
        '''
        (x_len, y_len) = self.numpy_dataset.shape
        return self.numpy_dataset.reshape(x_len * y_len)

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[region.x1:region.x2, region.y1:region.y2]
        return SpatialRegion(numpy_region_subset)

    def value_at(self, point):
        return self.numpy_dataset[point.x, point.y]

    def save(self, filename):
        '''
        Saves dataset to a file.
        '''
        ds_numpy = self.as_numpy
        np.save(filename, ds_numpy)
        self.logger.info('Saved to {}: {}'.format(filename, ds_numpy.shape))

    def apply_function_scalar(self, function_region_scalar):
        '''
        Applies an instance of FunctionRegionScalar on this region, to get a SpatialRegion
        as a result.

        Should be reusable in SpatioTemporalRegion (still returns a SpatialRegion), even though
        the value passed to the function is expected to be an array.

        Cannot be used inside the class iterator!
        '''

        # condition check: the 2D regions should have the same shape
        assert self.x_len == function_region_scalar.x_len
        assert self.y_len == function_region_scalar.y_len

        # condition passed, the output will have the same 2D shape
        # the output dtype is given by the function
        result_np = np.zeros((self.x_len, self.y_len), dtype=function_region_scalar.dtype)
        result = SpatialRegion(result_np)

        # use internal iterator! this means we can't use this function inside another iteration...
        for (point, value) in self:

            # get the function at the point (can vary!), apply it to get result at point
            function_at_point = function_region_scalar.function_at(point)

            # store result in numpy array: cannot use result.value_at(point)
            result_np[point.x, point.y] = function_at_point(value)

        return result

    def apply_function_series(self, function_region_series):
        '''
        Applies an instance of FunctionRegionSeries on this region, to get a
        SpatioTemporalRegion as a result.

        Cannot be used inside the class iterator!
        '''

        # condition check: the 2D regions should have the same shape
        assert self.x_len == function_region_series.x_len
        assert self.y_len == function_region_series.y_len

        # the length and dtype of the result series is given by the function
        result_np = np.zeros((function_region_series.output_len, self.x_len, self.y_len),
                             dtype=function_region_series.dtype)

        # ugly import to avoid circular imports
        from .temporal import SpatioTemporalRegion
        result = SpatioTemporalRegion(result_np)

        # use internal iterator! this means we can't use this function inside another iteration...
        for (point, series) in self:

            # get the function at the point (can vary!), apply it to get result at point
            function_at_point = function_region_series.function_at(point)

            # store result in numpy array
            result_np[:, point.x, point.y] = function_at_point(series)

        return result

    @classmethod
    def create_from_1d(cls, list_1d, x, y):
        '''
        Given a list, reshapes the elements to form a 2d region with (x, y) shape.
        '''
        numpy_dataset = reshape_1d_to_2d(list_1d, x, y)
        return SpatialRegion(numpy_dataset)

    @classmethod
    def region_with_zeroes(cls, region):
        '''
        Creates a spatial region filled with zeroes, with the dimensions given by the region.
        Notice that indices will start at (0, 0)
        '''
        (x_len, y_len) = (Region.x2 - Region.x1, Region.y2 - Region.y1)
        zero_array = np.zeros((x_len, y_len))
        return SpatialRegion(zero_array)

    def __str__(self):
        return str(self.numpy_dataset)


if __name__ == '__main__':

    # test iteration
    data2d = np.arange(15).reshape((3, 5))
    print(data2d)
    sp_region = SpatialRegion(data2d)
    for (point, value) in sp_region:
        print('Value at {} = {}'.format(point, value))

    # iterate again?
    for (point, value) in sp_region:
        print('Value at {} = {}'.format(point, value))
