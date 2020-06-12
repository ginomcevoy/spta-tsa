import numpy as np

# this approach avoids circular imports
import spta.region.base
import spta.region.spatial
import spta.region.temporal


class FunctionRegion(spta.region.spatial.SpatialRegion):
    '''
    A SpatialRegion where the value at each point is a function that can be applied to another
    region (SpationTemporalRegion instance?).

    This implementation reifies the Visitor pattern: the domain is the visitor, and the
    functions get "visited" by the domain so that the function can be applied at each point.

    The constructor accepts an optional dtype, which will be used when creating the output.
    '''
    def __init__(self, numpy_dataset, dtype=np.float64):
        super(FunctionRegion, self).__init__(numpy_dataset)
        self.dtype = dtype

    def function_at(self, point):
        '''
        Returns the function at the specified point.
        '''
        # reuse the numpy implementation, assume that there is a function at the point instead
        # of a single value.
        function_at_point = self.value_at(point)

        # this checks the assumption above...
        assert callable(function_at_point)

        return function_at_point

    def apply_to(self, domain_region):
        '''
        Apply this function region to a domain region.
        Subclasses should implement this by declaring how the function "accepts" the domain
        visitor in the Visitor pattern.
        '''
        raise NotImplementedError


class FunctionRegionScalar(FunctionRegion):
    '''
    A FunctionRegion that returns a scalar value for a each point in the parameter region.
    The result should be a SpatialRegion.
    '''

    def apply_to(self, domain_region):
        '''
        Apply this function to a domain region, to get a SpatialRegion as result.

        Lets the parameter region handle the call by default. This enables outputs that are
        subclasses of SpatialRegion, without the function knowing about the polymorphism.
        '''
        # the function passes itself, delegates computation to the region
        # visitor pattern: visitor.visitElementA(this)
        result_region = domain_region.apply_function_scalar(self)

        # should not be SpatioTemporalRegion, otherwise we should have used a different function
        assert isinstance(result_region, spta.region.spatial.SpatialRegion)
        assert not isinstance(result_region, spta.region.temporal.SpatioTemporalRegion)

        return result_region


class FunctionRegionScalarSame(FunctionRegionScalar):
    '''
    Function region that applies, at every point, the same provided function returning a scalar.
    The region is built based on provided x_len and y_len.
    '''
    def __init__(self, function, x_len, y_len, dtype=np.float64):

        # create the numpy dataset by reshaping an array of functions
        function_list = [
            function
            for i in range(0, x_len * y_len)
        ]
        function_np = np.array(function_list).reshape((x_len, y_len))
        super(FunctionRegionScalarSame, self).__init__(function_np, dtype)


class FunctionRegionSeries(FunctionRegion):
    '''
    A FunctionRegion that returns a series for each point in the parameter region.
    The result should be a SpatioTemporalRegion (!)
    '''
    def __init__(self, numpy_dataset, dtype=np.float64):
        super(FunctionRegionSeries, self).__init__(numpy_dataset, dtype)

    def apply_to(self, domain_region, output_len):
        '''
        Apply this function to a domain region, to get a SpatioTemporalRegion as result.
        The length of the resulting series must be known in order to create the numpy object.

        Lets the parameter region handle the call by default. This enables outputs that are
        subclasses of SpatioTemporalRegion, without the function knowing about the polymorphism.
        '''
        # the function passes itself, delegates computation to the region
        # visitor pattern: visitor.visitElementB(this)
        result_region = domain_region.apply_function_series(self, output_len)

        # should be SpatioTemporalRegion otherwise we should have used a different function
        assert isinstance(result_region, spta.region.temporal.SpatioTemporalRegion)

        return result_region


class FunctionRegionSeriesSame(FunctionRegionSeries):
    '''
    Function region that applies, at every point, the same provided function returning a series.
    The region is built based on provided x_len and y_len.
    '''
    def __init__(self, function, x_len, y_len, dtype=np.float64):

        # create the numpy dataset by reshaping an array of functions
        function_list = [
            function
            for i in range(0, x_len * y_len)
        ]
        function_np = np.array(function_list).reshape((x_len, y_len))
        super(FunctionRegionSeriesSame, self).__init__(function_np, dtype)
