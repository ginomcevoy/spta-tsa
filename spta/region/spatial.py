import numpy as np

from spta.util import log as log_util

from . import Point, Region, reshape_1d_to_2d


# model + test + forecast -> error

# arimitas -> error de cada arima en su entrenamiento
# 1 model + test/region + forecast/region -> error en cada punto

class SpatialRegion(log_util.LoggerMixin):

    def __init__(self, numpy_dataset, region_metadata=None):
        super(SpatialRegion, self).__init__()

        self.numpy_dataset = numpy_dataset
        self.metadata = region_metadata

        # used for iterating over points
        self.point_index = 0

        # convenience
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

    def empty_region_2d(self):
        '''
        Returns an empty SpatialRegion with the same shape as this region.
        '''
        empty_region_np = np.empty((self.x_len, self.y_len))
        return SpatialRegion(empty_region_np)

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


class SpatialRegionDecorator(SpatialRegion):
    '''
    Reifies the decorator pattern for a SpatialRegion.
    Decorates can extend the behavior of another SpatialRegion instance.
    '''
    def __init__(self, decorated_region, **kwargs):

        # keep SpatialRegion initialization (shape, iterator index, etc)
        self.decorated_region = decorated_region
        super(SpatialRegionDecorator, self).__init__(decorated_region.numpy_dataset, **kwargs)

    def __next__(self):
        return self.decorated_region.__next__()

    def region_subset(self, region):
        return self.decorated_region.region_subset(region)

    def value_at(self, point):
        return self.decorated_region.value_at(point)

    def empty_region_2d(self):
        return self.decorated_region.empty_region_2d()

    def save(self, filename):
        return self.decorated_region.save(filename)

    def apply_function_scalar(self, function_region_scalar):
        return self.decorated_region.apply_function_scalar(function_region_scalar)

    def apply_function_series(self, function_region_series):
        return self.decorated_region.apply_function_series(function_region_series)


class SpatialCluster(SpatialRegionDecorator):
    '''
    A subset of a spatial region that represents a cluster created by a clustering
    algorithm, or obtained by interacting with a spatio-temporal cluster.

    Implemented by decorating an existing SpatialRegion with cluster data.
    See temporal.SpatioTemporalCluster for more information.
    '''
    def __init__(self, decorated_region, spatial_mask, label=1, region_metadata=None):

        if region_metadata is not None:
            error_msg = 'region_metadata not supported for {}!'
            raise NotImplementedError(error_msg.format(self.__class__.__name__))

        super(SpatialCluster, self).__init__(decorated_region)

        # cluster-specific
        self.spatial_mask = spatial_mask
        self.label = label

    def region_subset(self, region):
        error_msg = 'region_subset not allowed for {}!'
        raise NotImplementedError(error_msg.format(self.__class__.__name__))

    def value_at(self, point):
        if self.spatial_mask.value_at(point):
            return self.decorated_region.value_at(point)
        else:
            raise ValueError('Point not in cluster mask: {}'.format(point))

    def empty_region(self):
        '''
        Returns an empty SpatialCluster with the same shape as this region, and same mask/label.
        '''
        empty_spatial_region = self.decorated_region.empty_region()
        return SpatialCluster(decorated_region=empty_spatial_region,
                              spatial_mask=self.spatial_mask, label=self.label)

    def __next__(self):
        '''
        Used for iterating over points in the cluster. Only points in the mask are iterated!
        The iterator returns the tuple (Point, value) for each point.
        '''
        while True:

            # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
            if self.point_index >= self.y_len * self.x_len:
                # stop iteration, but allow reuse of iterator from start again
                self.point_index = 0
                raise StopIteration

            # candidate point, first is (0, 0)
            i = int(self.point_index / self.y_len)
            j = self.point_index % self.y_len
            point_i_j = Point(i, j)

            # next point to evaluate after this
            self.point_index += 1

            if self.spatial_mask.value_at(point_i_j):
                # found point in the mask
                break

        # return next point in the mask
        return (point_i_j, self.value_at(point_i_j))

    def apply_function_scalar(self, function_region_scalar):
        '''
        Applies an instance of FunctionRegionScalar on this region, to get a SpatialCluster
        as a result.

        Behaves similar to the SpatialRegion implementation, with these differences:
          - only points in the mask are iterated (iterator is overridden)
          - a new SpatialCluster is created, instead of a new SpatialRegion.

        Should be reusable in SpatioTemporalCluster (still returns a SpatialCluster), even though
        the value passed to the function is expected to be an array.

        Cannot be used inside the class iterator!
        '''

        # call the parent code (SpatialRegion)
        # since the iterator is overridden, this should only iterate over points in the mask
        spatial_region = super(SpatialCluster, self).apply_function_scalar(function_region_scalar)

        # return a SpatialCluster instead!
        # Notice that the calling function is not aware of the change
        return SpatialCluster(spatial_region, self.spatial_mask, self.label)

    def apply_function_series(self, function_region_series):
        '''
        Applies an instance of FunctionRegionSeries on this region, to get a
        SpatioTemporalCluster as a result.

        Behaves similar to the SpatialRegion implementation, with these differences:
          - only points in the mask are iterated (iterator is overridden)
          - a new SpatioTemporalCluster is created, instead of a new SpatioTemporalRegion.

        Cannot be used inside the class iterator!
        '''

        # call the parent code (SpatialRegion)
        # since the iterator is overridden, this should only iterate over points in the mask
        spt_region = super(SpatialCluster, self).apply_function_series(function_region_series)

        # return a SpatioTemporalCluster instead!
        # Notice that the calling function is not aware of the change
        from .temporal import SpatioTemporalCluster
        return SpatioTemporalCluster(spt_region, self.spatial_mask, self.label)


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
