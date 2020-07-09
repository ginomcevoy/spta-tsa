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

    def new_spatial_region(self, numpy_dataset):
        '''
        Creates a new instance of SpatialRegion with the underlying numpy data.
        '''
        return SpatialRegion(numpy_dataset)

    def new_spatio_temporal_region(self, numpy_dataset):
        '''
        Creates a new instance of SpatioTemporalRegion with the underlying numpy data.
        '''
        # ugly import to avoid circular imports
        from .temporal import SpatioTemporalRegion
        return SpatioTemporalRegion(numpy_dataset)

    def apply_function_scalar(self, function_region_scalar):
        '''
        Applies an instance of FunctionRegionScalar on this region, to get a SpatialRegion
        as a result.

        Should be reusable in SpatioTemporalRegion (still returns a SpatialRegion), even though
        the value passed to the function is expected to be an array.

        If the iterator is overridden, the behavior may change, e.g. for clusters.
        Cannot be used inside the class iterator!
        '''

        # condition check: the 2D regions should have the same shape
        assert self.x_len == function_region_scalar.x_len
        assert self.y_len == function_region_scalar.y_len

        # condition passed, the output will have the same 2D shape
        # the output dtype is given by the function
        result_np = np.zeros((self.x_len, self.y_len), dtype=function_region_scalar.dtype)

        # use internal iterator! this means we can't use this function inside another iteration...
        for (point, value) in self:

            # self.logger.debug('Iterating in {} at point {}'.format(self, point))

            # get the function at the point (can vary!), apply it to get result at point
            function_at_point = function_region_scalar.function_at(point)

            # store result in numpy array: cannot use result.value_at(point)
            result_np[point.x, point.y] = function_at_point(value)

        # create a new instance of SpatialRegion
        # the call may be polymorphic resulting in instances of other child classes.
        return self.new_spatial_region(result_np)

    def apply_function_series(self, function_region_series, output_len):
        '''
        Applies an instance of FunctionRegionSeries on this region, to get a
        SpatioTemporalRegion as a result.

        If the iterator is overridden, the behavior may change, e.g. for clusters.
        Cannot be used inside the class iterator!
        '''

        # condition check: the 2D regions should have the same shape
        assert self.x_len == function_region_series.x_len
        assert self.y_len == function_region_series.y_len

        # the length and dtype of the result series is given by the function
        result_np = np.zeros((output_len, self.x_len, self.y_len),
                             dtype=function_region_series.dtype)

        # use internal iterator! this means we can't use this function inside another iteration...
        for (point, series) in self:

            # get the function at the point (can vary!), apply it to get result at point
            function_at_point = function_region_series.function_at(point)

            # store result in numpy array
            result_np[:, point.x, point.y] = function_at_point(series)

        # create a new instance of SpatialRegion
        # the call may be polymorphic resulting in instances of other child classes.
        return self.new_spatio_temporal_region(result_np)


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
        return self.new_spatial_region(numpy_region_subset)

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
        return self.new_spatial_region(repeated_value_np)

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

    def new_spatial_region(self, numpy_dataset):
        '''
        By default, delegate SpatialRegion instance to decorated region using Chain of
        Responsibility pattern.
        '''
        return self.decorated_region.new_spatial_region(numpy_dataset)

    def new_spatio_temporal_region(self, numpy_dataset):
        '''
        By default, delegate SpatioTemporalRegion instance to decorated region using Chain of
        Responsibility pattern.
        '''
        return self.decorated_region.new_spatio_temporal_region(numpy_dataset)

    def region_subset(self, region):
        return self.decorated_region.region_subset(region)

    def value_at(self, point):
        return self.decorated_region.value_at(point)

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

    No need to override these, because they will use new_spatial_region correctly:
        - repeat_value
        - repeat_point
        - empty_region_2d
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

    def new_spatial_region(self, numpy_dataset):
        '''
        Creates a new instance of SpatialCluster with the same partition and cluster index.
        This will wrap a new decorated region, that was built by the original decorator
        using Chain of Responsibility pattern.
        '''
        new_decorated_region = self.decorated_region.new_spatial_region(numpy_dataset)
        return SpatialCluster(decorated_region=new_decorated_region,
                              partition=self.partition,
                              cluster_index=self.cluster_index)

    def new_spatio_temporal_region(self, numpy_dataset):
        # We need a decorated SpatioTemporalRegion, which we don't have.
        raise NotImplementedError

    def region_subset(self, region):
        '''
        TODO don't need this yet
        '''
        error_msg = 'region_subset not allowed for {}!'
        raise NotImplementedError(error_msg.format(self.__class__.__name__))

    def value_at(self, point):
        '''
        Returns the value at the point, only if the point belongs to this cluster.
        '''
        # sanity check
        if point is None:
            return np.nan

        if self.partition.is_member(point, self.cluster_index):
            return self.decorated_region.value_at(point)
        else:
            raise ValueError('Point not in cluster: {}'.format(point))

    def __next__(self):
        '''
        Used for iterating *only* over points in the cluster. Points not in the cluster are skipped
        by this iterator!
        The iterator returns the tuple (Point, value) for each point.

        This iterator will also play a role in apply_function_scalar and apply_function_series,
        the function will be applied only the points that belong to this cluster.
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
