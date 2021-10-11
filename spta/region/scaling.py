'''
This module handles scale and descale of spatio-temporal regions.
'''

import numpy as np

from .function import FunctionRegionScalarSame, FunctionRegionSeries, FunctionRegionSeriesSame
from .temporal import SpatioTemporalDecorator


class ScaleFunction(FunctionRegionSeriesSame):
    '''
    A function region that scales the series of each point of a spatio-temporal region, so that
    values fall between [0, 1]. The result is an instance of SpatioTemporalScaled.

    To create a scaled version of the spatio-temporal region spt_region:

    scale_function = ScaleFunction()
    scaled_region = scale_function.apply_to(spt_region, series_len)

    The scaling occurs independently for each point: each series will be scaled to be in the
    interval [0, 1]. To revert the process, the mininum and maximum values are stored for each
    point, the relevant data is stored in two spatial regions (scale_min and
    scale_max) inside the scaled spatio-temporal region. The process can be reverted
    by calling descale() on the region. This method will only work if the data was scaled
    at some point, but it should work on decorated versions, e.g. a cluster of a scaled region.
    '''

    def __init__(self, x_len, y_len):
        # This function region does not have a relevant underlying numpy in it.
        # We still need the proper shape though
        super(ScaleFunction, self).__init__(None, x_len, y_len)

    def function_at(self, point):

        # retrieve the min/max values for this point
        series_min = self.scale_min.value_at(point)
        series_max = self.scale_max.value_at(point)

        # this function scales the series at each point independently of other points
        def scale_this_point(series):

            # sanity checks
            if np.isnan(series_min) or np.isnan(series_max):
                return np.repeat(np.nan, repeats=self.series_len)

            if series_min == series_max:
                return np.zeros(self.series_len)

            # scale here
            return (series - series_min) / (series_max - series_min)

        # the function to be applied at this point
        return scale_this_point

    def apply_to(self, spt_region, output_len):

        (series_len, x_len, y_len) = spt_region.shape

        # restriction to ensure proper behavior
        assert series_len == output_len

        # save the series_len
        # TODO get rid of this, always save output_len in FunctionRegionSeries
        self.series_len = series_len

        # calculate the min, max values for each series
        # we will save these outputs to allow descale
        minFunction = FunctionRegionScalarSame(np.nanmin, x_len, y_len)
        self.scale_min = minFunction.apply_to(spt_region)

        maxFunction = FunctionRegionScalarSame(np.nanmax, x_len, y_len)
        self.scale_max = maxFunction.apply_to(spt_region)

        # call normal function behavior, function_at will be retrieved at each point of
        # the region for application.
        region_with_scaled_dataset = \
            super(ScaleFunction, self).apply_to(spt_region, output_len)

        # save the original metadata if available
        # TODO ensure that the metadata now says 'scaled', this requires more thought
        # because we need another copy of the metadata
        region_with_scaled_dataset.region_metadata = spt_region.region_metadata

        # now create an instance of SpatioTemporalScaled, which has data for scale
        scaled_region = SpatioTemporalScaled(region_with_scaled_dataset=region_with_scaled_dataset,
                                             scale_min=self.scale_min,
                                             scale_max=self.scale_max,
                                             region_metadata=spt_region.region_metadata)
        return scaled_region


class SpatioTemporalScaled(SpatioTemporalDecorator):
    '''
    A decorator of SpatioTemporalRegion that holds, for each point, the series scaled to

    A decorator that scales the temporal series by scaling all the series to values
    between 0 and 1. This can be achieved by applying the formula to each series:

    scaled_series = (series - min) / (max - min)

    We store the min and max values, so that the series can be later recovered by using:

    series = scaled_series (max - min) + min

    The min and max values are valid for each series. These are stored in spatial regions
    called scale_min and scale_max, respectively.

    If min = max, then scaled is fixed to 0.
    If the series is Nan, save the series, and set min = max = 0.
    '''
    def __init__(self, region_with_scaled_dataset, scale_min, scale_max,
                 **kwargs):

        # self.logger.debug('SpatioTemporalScaled kwargs: {}'.format(kwargs))

        # resolve the diamond problem using super
        # metadata will be saved here as part of kwargs
        super(SpatioTemporalScaled, self).__init__(region_with_scaled_dataset, **kwargs)

        # the data is already scaled, so just store the variables
        self.scale_min = scale_min
        self.scale_max = scale_max

    def new_spatio_temporal_region(self, numpy_dataset):
        '''
        Create a new scaled region with the provided dataset, keeping decorated behavior.
        '''
        # self.logger.debug('Called SpatioTemporalScaled.new_spatio_temporal_region()')
        # We still want to keep the decorated behavior, so ask the decorated region to create a
        # new instance, and wrap it properly for a scaled region.
        new_decorated_region = self.decorated_region.new_spatio_temporal_region(numpy_dataset)

        return SpatioTemporalScaled(region_with_scaled_dataset=new_decorated_region,
                                    scale_min=self.scale_min,
                                    scale_max=self.scale_max,
                                    region_metadata=self.region_metadata)

    def region_subset(self, region):
        '''
        If asked for a subset, then also subset the scaling data.
        '''
        # this will NOT return a SpatioTemporalScaled because it will use the
        # new_spatio_temporal_region() method of the decorated region!
        subset_region = super(SpatioTemporalScaled, self).region_subset(region)
        self.logger.debug('subset_region in SpatioTemporalScaled: {!r}'.format(subset_region))

        # get the scaling data appropriate for this subset
        scale_min_subset = self.scale_min.region_subset(region)
        scale_max_subset = self.scale_max.region_subset(region)

        # fix the scaling here, use the subset_region which was obtained from the decorated region,
        # we won't end up with two layers of scaling decoration.
        return SpatioTemporalScaled(region_with_scaled_dataset=subset_region,
                                    scale_min=scale_min_subset,
                                    scale_max=scale_max_subset)

    def has_scaling(self):
        '''
        Flag to indicate that the region can be descaled (yes here!)
        '''
        return True

    def descale(self):
        '''
        Revert the scale process. This will create a spatio-temporal region with the
        same properties as the (possibly decorated) region that was scaled.
        '''

        # implement this by creating a FunctionRegionSeries and applying it to this (self) region.
        class DescaleRegion(FunctionRegionSeriesSame):

            def __init__(function_self, x_len, y_len):
                # This function region does not have a relevant underlying numpy in it.
                # We still need the proper shape though
                super(DescaleRegion, function_self).__init__(None, x_len, y_len)

            def function_at(function_self, point):
                # Here, we descale the series at the point using the scale data
                # available in the region. Notice the use of "self" (SpatioTemporalScaled).
                min_for_point = self.scale_min.value_at(point)
                max_for_point = self.scale_max.value_at(point)

                # log_msg = 'point {}: scale_min={}, scale_max={}'
                # self.logger.debug(log_msg.format(point, min_for_point, max_for_point))

                def descale(series):
                    # the descaling function that will be called on the series at each point
                    return (max_for_point - min_for_point) * series + min_for_point

                return descale

        # apply the function defined above
        # this will create a SpatioTemporalScaled region
        descale_function = DescaleRegion(self.x_len, self.y_len)
        descaled_region = descale_function.apply_to(self, self.series_len)

        # Drop the scale behavior now: we do this by reverting to an instance of region
        # decorated by this scale instance. The new instance does not have scale_* regions.
        # If there is a chain of decorated regions on top of each other, this should "remove" the
        # scale decoration but leave the others to maintain expected functionalities.
        region_without_scaling = \
            self.decorated_region.new_spatio_temporal_region(descaled_region.numpy_dataset)

        return region_without_scaling

    def save(self):
        '''
        In addition of saving the numpy dataset, also save the min/max regions.
        Requires the metadata!
        '''

        # save dataset
        super(SpatioTemporalScaled, self).save()

        # save min
        min_filename = self.region_metadata.scaled_min_filename
        np.save(min_filename, self.scale_min.numpy_dataset)
        self.logger.info('Saved scale_min to {}'.format(min_filename))

        # save max
        max_filename = self.region_metadata.scaled_max_filename
        np.save(max_filename, self.scale_max.numpy_dataset)
        self.logger.info('Saved scale_max to {}'.format(max_filename))

    def __next__(self):
        '''
        Don't use the default iterator here, which comes from SpatialDecorator.
        Instead, iterate like a spatio-temporal region, as the decorated region does.
        '''
        return self.decorated_region.__next__()
