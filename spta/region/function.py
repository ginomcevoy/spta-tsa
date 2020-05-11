from . import SpatialRegion, SpatioTemporalRegion


class FunctionRegion(SpatialRegion):
    '''
    A SpatialRegion where the value at each point is a function that can be applied to another
    region (SpationTemporalRegion instance?).
    '''

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

    def apply_to(self, spt_region):
        '''
        Apply this function region to a spatio-temporal region.
        Subclasses should implement this.
        '''
        raise NotImplementedError


class FunctionRegionScalar(FunctionRegion):
    '''
    A FunctionRegion that returns a scalar value for a each point in the parameter region.
    The result should be a SpatialRegion.
    '''

    def apply_to(self, spt_region):
        '''
        Apply this function to a spatio-temporal region, to get a SpatialRegion as result.

        Lets the parameter region handle the call by default. This enables outputs that are
        subclasses of SpatialRegion, without the function knowing about the polymorphism.
        '''
        # the function passes itself, delegates computation to the region
        result_region = spt_region.apply_function_scalar(self)

        # should not be SpatioTemporalRegion, otherwise we should have used a different function
        assert isinstance(result_region, SpatialRegion)
        assert not isinstance(result_region, SpatioTemporalRegion)

        return result_region


class FunctionRegionSeries(FunctionRegion):
    '''
    A FunctionRegion that returns a series for each point in the parameter region.
    The result should be a SpatioTemporalRegion (!)

    The length of the result should be known.
    '''
    def __init__(self, numpy_dataset, output_len):
        super(FunctionRegion, self).__init__(numpy_dataset)
        self.output_len = output_len

    def apply_to(self, spt_region):
        '''
        Apply this function to a spatio-temporal region, to get a SpatioTemporalRegion as result.

        Lets the parameter region handle the call by default. This enables outputs that are
        subclasses of SpatioTemporalRegion, without the function knowing about the polymorphism.
        '''
        # the function passes itself, delegates computation to the region
        result_region = spt_region.apply_function_series(self)

        # should be SpatioTemporalRegion otherwise we should have used a different function
        assert isinstance(result_region, SpatioTemporalRegion)

        return result_region
