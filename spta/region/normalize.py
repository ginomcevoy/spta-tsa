'''
This module handles normalization and denormalization of spatio-temporal regions.
'''

import numpy as np

from .function import FunctionRegionScalarSame
from .temporal import SpatioTemporalDecorator


class SpatioTemporalNormalized(SpatioTemporalDecorator):
    '''
    A decorator that normalizes the temporal series by scaling all the series to values
    between 0 and 1. This can be achieved by applying the formula to each series:

    normalized_series = (series - min) / (max - min)

    We store the min and max values, so that the series can be later recovered by using:

    series = normalized_series (max - min) + min

    The min and max values are valid for each series. These are stored in spatial regions
    called normalization_min and normalization_max, respectively.

    If min = max, then scaled is fixed to 0.
    If the series is Nan, save the series, and set min = max = 0.
    '''
    def __init__(self, decorated_region, **kwargs):

        self.logger.debug('SpatioTemporalNormalized kwargs: {}'.format(kwargs))

        (series_len, x_len, y_len) = decorated_region.shape

        # moved here to avoid circular imports between FunctionRegion and SpatioTemporalRegion
        from . import function as function_region

        # calculate the min, max values for each series
        # we will save these outputs to allow denormalization
        minFunction = FunctionRegionScalarSame(np.nanmin, x_len, y_len)
        self.normalization_min = minFunction.apply_to(decorated_region)

        maxFunction = FunctionRegionScalarSame(np.nanmax, x_len, y_len)
        self.normalization_max = maxFunction.apply_to(decorated_region)

        # this function normalizes the series at each point independently of other points
        def normalize(series):

            # calculate min/max (again...)
            series_min = np.nanmin(series)
            series_max = np.nanmax(series)

            # sanity checks
            if np.isnan(series_min) or np.isnan(series_max):
                return np.repeat(np.nan, repeats=series_len)

            if series_min == series_max:
                return np.zeros(series_len)

            # normalize here
            return (series - series_min) / (series_max - series_min)

        # call the function
        normalizing_function = function_region.FunctionRegionSeriesSame(normalize, x_len, y_len)
        normalized_region = normalizing_function.apply_to(decorated_region, series_len)
        normalized_region.region_metadata = decorated_region.region_metadata

        # use the normalized region here, all decorating functions from other decorators should be
        # applied to this normalized region instead
        super(SpatioTemporalNormalized, self).__init__(normalized_region, **kwargs)

    def save(self):
        '''
        In addition of saving the numpy dataset, also save the min/max regions.
        Requires the metadata!
        '''

        # save dataset
        super(SpatioTemporalNormalized, self).save()

        # save min
        min_filename = self.region_metadata.norm_min_filename
        np.save(min_filename, self.normalization_min.numpy_dataset)
        self.logger.info('Saved norm_min to {}'.format(min_filename))

        # save max
        max_filename = self.region_metadata.norm_max_filename
        np.save(max_filename, self.normalization_max.numpy_dataset)
        self.logger.info('Saved norm_max to {}'.format(max_filename))

    def __next__(self):
        '''
        Don't use the default iterator here, which comes from SpatialDecorator.
        Instead, iterate like a spatio-temporal region, as the decorated region does.
        '''
        return self.decorated_region.__next__()
