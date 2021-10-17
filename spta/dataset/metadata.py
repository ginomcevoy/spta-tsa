import numpy as np

from spta.util import log as log_util
from spta.util import maths as maths_util


class TimeToSeries(log_util.LoggerMixin):

    def __repr__():
        '''Representation used to build a filename for a dataset.'''
        raise NotImplementedError

    def convert(self, raw_dataset, time_to_series):
        '''Convert a dataset to a different sample frequency, or another temporal conversion.'''
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, TimeToSeries):
            return False

        return repr(self) == repr(other)


class SamplesPerDay(TimeToSeries):
    '''A representation for series that indicates a frequency of samples per day.'''

    def __init__(self, samples_per_day):
        self.samples_per_day = samples_per_day

    def convert(self, numpy_dataset, time_to_series):
        '''
        Given a numpy spatio-temporal dataset with a known frequency of samples per day (e.g. spd=4),
        this method may convert the points in each day to get 1 point per day (spd = 1).

        Assumes [series:x:y] shape for the dataset.
        '''
        # can only convert another SamplesPerDay
        if not hasattr(time_to_series, 'samples_per_day'):
            raise ValueError('time_to_series should be instance of SamplesPerDay')

        if self.samples_per_day == time_to_series.samples_per_day:
            # nothing to do
            return numpy_dataset

        # assume convenient things for now, for example that we are using a dataset
        # with spd=4 and that we want spd=1.
        # TODO refine this in the future
        assert self.samples_per_day == 4
        assert time_to_series.samples_per_day == 1

        # we have 4 points per day
        # average these four points to get a smoother curve
        (series_len, x_len, y_len) = numpy_dataset.shape

        new_series_len = int(series_len / 4)
        single_point_per_day = np.empty((new_series_len, x_len, y_len))

        for x in range(0, x_len):
            for y in range(0, y_len):
                point_series = numpy_dataset[:, x, y]
                series_reshape = (new_series_len, 4)
                smooth = np.mean(np.reshape(point_series, series_reshape), axis=1)
                # sptr.log.debug('smooth: %s' % smooth)
                single_point_per_day[:, x, y] = np.array(smooth)

        log_msg = 'Reshaped {} (4spd) -> {} (1spd)'
        self.logger.info(log_msg.format(numpy_dataset.shape, single_point_per_day.shape))

        return single_point_per_day

    def __repr__(self):
        return '{}spd'.format(self.samples_per_day)


class AveragePentads(TimeToSeries):
    '''A representation for series that indicates that the data has been processed into average pentads.'''

    def convert(self, numpy_dataset, time_to_series):
        '''
        For now, we don't convert average pentads, we assume that the numpy_dataset has
        already been processed externally.
        '''
        # only works on average pentads
        assert time_to_series.__class__.__name__ == 'AveragePentads'

        # do nothing
        return numpy_dataset

    def __repr__(self):
        return 'avg_pentads'


class TemporalMetadata(log_util.LoggerMixin):
    '''
    A metadata for raw datasets that deals with the temporal aspect.
    This class can be used to represent the whole dataset, or with a temporal slice.

    When applied to the whole dataset, it has a year of start of samples and a year
    for end of samples. It also uses an instance of TimeToSeries to indicate how the
    samples relate to real time.

    An instance of this class can be created when asked to retrieve a temporal slice
    of the whole dataset, in this case the sample frequency may be different.

    A dataset can compare its own metadata with the metadata of a requested slice,
    in order to produce the slice as a numpy_dataset.
    '''

    def __init__(self, year_start, year_end, time_to_series):
        self.year_start = year_start
        self.year_end = year_end
        self.time_to_series = time_to_series

    def years_to_series_interval(self, year_start_request, year_end_request):
        '''
        Converts a request of year interval to a series interval. To be able to do this,
        we need to know how the temporal information in the dataset relates to real dates,
        this is provided by the time_to_series of the dataset.

        Currently, this is only possible to do for SamplesPerDay.
        '''
        # limitation for now
        if not hasattr(self.time_to_series, 'samples_per_day'):
            raise ValueError('time_to_series should be instance of SamplesPerDay')

        # check boundaries and sanity
        bad_request = year_start_request > year_end_request
        bad_start_request = year_start_request < self.year_start
        bad_end_request = year_end_request > self.year_end
        if bad_request or bad_start_request or bad_end_request:
            raise ValueError('Invalid year interval: {}-{}'.format(year_start_request, year_end_request))

        # this transforms years to samples
        (series_start, series_end) = \
            maths_util.years_to_series_interval(year_start=year_start_request,
                                                year_end=year_end_request,
                                                first_year_in_sample=self.year_start,
                                                samples_per_day=self.time_to_series.samples_per_day)

        return (series_start, series_end)

    def convert_time_to_series(self, numpy_dataset, time_to_series):
        '''See TimeToSeries and its subclasses to which this is delegated.'''
        return self.time_to_series.convert(numpy_dataset, time_to_series)

    def __repr__(self):
        '''
        A representation of this temporal metadata, can be used to find files representing
        temporal slices of a dataset.
        '''
        return '{}_{}_{!r}'.format(self.year_start, self.year_end, self.time_to_series)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, TemporalMetadata):
            return False

        same_start = self.year_start == other.year_start
        same_end = self.year_end == other.year_end
        same_time_to_series = self.time_to_series == other.time_to_series

        return same_start and same_end and same_time_to_series
