import numpy as np

from . import base
from . import Region, reshape_1d_to_2d

from spta.util import arrays as arrays_util


class DomainRegion(base.BaseRegion):
    '''
    A region that represents the domain of a function. This is a parent class for both
    SpatialRegion and SpatioTemporalRegion.
    '''

    def __init__(self, numpy_dataset, **kwargs):
        super(DomainRegion, self).__init__(numpy_dataset)

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

            # self.logger.debug('Iterating in {} at point {}'.format(self, point))

            # get the function at the point (can vary!), apply it to get result at point
            function_at_point = function_region_scalar.function_at(point)

            # store result in numpy array: cannot use result.value_at(point)
            result_np[point.x, point.y] = function_at_point(value)

        return result

    def apply_function_series(self, function_region_series, output_len):
        '''
        Applies an instance of FunctionRegionSeries on this region, to get a
        SpatioTemporalRegion as a result.

        Cannot be used inside the class iterator!
        '''

        # condition check: the 2D regions should have the same shape
        assert self.x_len == function_region_series.x_len
        assert self.y_len == function_region_series.y_len

        # the length and dtype of the result series is given by the function
        result_np = np.zeros((output_len, self.x_len, self.y_len),
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


class SpatialRegion(DomainRegion):
    '''
    A 2-dimensional spatial region, backed by a 2-d numpy dataset.
    Can be applied to FunctionRegion subclasses.
    '''

    def __init__(self, numpy_dataset, **kwargs):
        super(SpatialRegion, self).__init__(numpy_dataset)

    def __next__(self):
        '''
        Used for iterating over points.
        The iterator returns the tuple (Point, value) for each point.
        '''

        # use the base iterator to get next point
        next_point = super(SpatialRegion, self).__next__()

        # tuple output
        return (next_point, self.value_at(next_point))

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
        # sanity check
        if point is None:
            return np.nan

        return self.numpy_dataset[point.x, point.y]

    def find_minimum(self):
        '''
        Find the point with the minimum value, return the (point, value) tuple.

        Cannot be used inside another iterator of this instance!
        '''
        # save the min value
        min_value = np.Inf
        min_point = None

        # use the iterator, should work as expected for subclasses of SpatialRegion
        for (point, value) in self:
            if value < min_value:
                min_value = value
                min_point = point

        return (min_point, min_value)

    def find_maximum(self):
        '''
        Find the point with the maximum value, return the (point, value) tuple.

        Cannot be used inside another iterator of this instance!
        '''
        # save the max value
        max_value = -np.Inf
        max_point = None

        # use the iterator, should work as expected for subclasses of SpatialRegion
        for (point, value) in self:
            if value > max_value:
                max_value = value
                max_point = point

        return (max_point, max_value)

    def repeat_value(self, value):
        '''
        Creates a new spatial region with this same shape, where all the values are
        the same provided values.
        '''
        repeated_value_np = arrays_util.copy_value_as_matrix_elements(value, self.x_len,
                                                                      self.y_len)
        return SpatialRegion(repeated_value_np)

    def repeat_point(self, point):
        '''
        Creates a new spatial region with this same shape, where all the values are the
        same value as in the provided point.
        '''
        # reuse repeat_value
        value_at_point = self.value_at(point)
        return self.repeat_value(value_at_point)

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
        '''
        A string representation: if a name is available, return it.
        Otherwise, return a generic name.
        '''
        if hasattr(self, 'name'):
            return self.name
        else:
            return 'SpatialRegion {}'.format(self.shape)


class SpatialDecorator(SpatialRegion):
    '''
    Reifies the decorator pattern for a SpatialRegion.
    Decorates can extend the behavior of another SpatialRegion instance.
    '''
    def __init__(self, decorated_region, **kwargs):

        # keep SpatialRegion initialization (shape, iterator index, etc)
        self.decorated_region = decorated_region
        super(SpatialDecorator, self).__init__(decorated_region.numpy_dataset, **kwargs)

    def region_subset(self, region):
        return self.decorated_region.region_subset(region)

    def value_at(self, point):
        return self.decorated_region.value_at(point)

    def empty_region_2d(self):
        return self.decorated_region.empty_region_2d()

    def save_to(self, filename):
        return self.decorated_region.save_to(filename)

    def __str__(self):
        return str(self.decorated_region)

    # code below is wrong, we don't want to use the decorated function which will use the
    # decorated iterator, instead of a more useful iterator in decorating overrides

    # def apply_function_scalar(self, function_region_scalar):
    #     return self.decorated_region.apply_function_scalar(function_region_scalar)

    # def apply_function_series(self, function_region_series):
    #     return self.decorated_region.apply_function_series(function_region_series)


class SpatialCluster(SpatialDecorator):
    '''
    A subset of a spatial region that represents a cluster, created by a clustering
    algorithm, or obtained by interacting with a spatio-temporal cluster.

    The cluster behaves similar to a full spatial region when it comes to iteration and
    and some properties. Specifically:

    - Each cluster retains the shape of the entire region.
    - The iteration of points is made only over the points indicated by the partition and index
        (points belonging to the cluster).
    - A FunctionRegion can be applied to the cluster, only points in the mask will be considered
        (it uses the iteration above).
    - Region subsets are not allowed (TODO?), calling region_subset raises NotImplementedError
    - Attempt to use value_at at a point that is not a member will raise ValueError.

    Implemented by decorating an existing SpatialRegion with cluster data.
    See PartitionRegion for more information.
    '''
    def __init__(self, decorated_region, partition, cluster_index, **kwargs):
        '''
        decorated_region
            a spatial region that is being decorated with cluster behavior

        partition
            obtained with a clustering algorithm, it is capable of indicating which points
            belong to a cluster. It knows the number of clusters (k)

        cluster_index
            identifies this cluster in the partition, must be an integer in [0, k-1].
        '''
        super(SpatialCluster, self).__init__(decorated_region, **kwargs)

        # cluster-specific
        self.partition = partition
        self.cluster_index = cluster_index

    @property
    def cluster_len(self):
        '''
        The size of the cluster (# of points).
        '''
        # ask the mask for the size
        return self.partition.cluster_len(self.cluster_index)

    def region_subset(self, region):
        error_msg = 'region_subset not allowed for {}!'
        raise NotImplementedError(error_msg.format(self.__class__.__name__))

    def value_at(self, point):
        # sanity check
        if point is None:
            return np.nan

        if self.partition.is_member(point, self.cluster_index):
            return self.decorated_region.value_at(point)
        else:
            raise ValueError('Point not in cluster mask: {}'.format(point))

    def repeat_value(self, value):
        '''
        Creates a new spatial cluster with this same shape, where all the values are the
        same as the provided value.

        The value is repeated over all points for simplicity, but calling value_at on points
        outside the cluster should remain forbidden.

        NOTE: no need to reimplement repeat_point, it should correctly use this method
        polymorphically to create a SpatialCluster.
        '''
        repeated_region = self.decorated_region.repeat_value(value)
        return SpatialCluster(repeated_region, self.partition, self.cluster_index)

    def empty_region_2d(self):
        '''
        Returns an empty SpatialCluster with the same shape and same clustering info.
        '''
        empty_spatial_region = self.decorated_region.empty_region_2d()
        return SpatialCluster(decorated_region=empty_spatial_region,
                              partition=self.partition,
                              cluster_index=self.cluster_index)

    def __next__(self):
        '''
        Used for iterating *only* over points in the cluster. Points not in the cluster are skipped
        by this iterator!
        The iterator returns the tuple (Point, value) for each point.
        '''
        while True:

            # use the base iterator to get next candidate point in region
            # using DomainRegion instead of BaseRegion to avoid an import...
            try:
                candidate_point = DomainRegion.__next__(self)
            except StopIteration:
                # self.logger.debug('Base region iteration stopped')

                # when the base iterator stops, we also stop
                raise

            # self.logger.debug('{}: candidate = {}'.format(self, candidate_point))

            if self.partition.is_member(candidate_point, self.cluster_index):
                # found a member of the cluster
                next_point_in_cluster = candidate_point
                break

            # the point was not in the cluster, try with next candidate

        next_value = self.value_at(next_point_in_cluster)
        return (next_point_in_cluster, next_value)

    def apply_function_scalar(self, function_region_scalar):
        '''
        Applies an instance of FunctionRegionScalar on this cluster region, to get a SpatialCluster
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
        self.logger.debug('{} apply_function_scalar'.format(self))
        spatial_region = super(SpatialCluster, self).apply_function_scalar(function_region_scalar)

        # return a SpatialCluster instead!
        # Notice that the calling function is not aware of the change
        # (visitor is not aware of how the visited element is of a different subclass)
        return SpatialCluster(spatial_region, self.partition, self.cluster_index)

    def apply_function_series(self, function_region_series, output_len):
        '''
        Applies an instance of FunctionRegionSeries on this cluster region, to get a
        SpatioTemporalCluster as a result.

        Behaves similar to the SpatialRegion implementation, with these differences:
          - only points in the mask are iterated (iterator is overridden)
          - a new SpatioTemporalCluster is created, instead of a new SpatioTemporalRegion.

        Cannot be used inside the class iterator!
        '''
        self.logger.debug('{} apply_function_series'.format(self))

        # call the parent code (SpatialRegion)
        # since the iterator is overridden, this should only iterate over points in the mask
        spt_region = super(SpatialCluster, self).apply_function_series(function_region_series,
                                                                       output_len)

        # return a SpatioTemporalCluster instead!
        # Notice that the calling function is not aware of the change
        # (visitor is not aware of how the visited element is of a differnt subclass)
        from .temporal import SpatioTemporalCluster
        return SpatioTemporalCluster(spt_region, self.partition, self.cluster_index)

    def __str__(self):
        '''
        A string representation: if a name is available, return it.
        Otherwise, return a generic name.
        '''
        if hasattr(self, 'name'):
            return self.name
        else:
            return 'SpatialCluster {}'.format(self.shape)


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
