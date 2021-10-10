import numpy as np

from spta.util import log as log_util


class TimeToSeries(log_util.LoggerMixin):

    def __repr__():
        '''Representation used to build a filename for a dataset.'''
        raise NotImplementedError

    def convert(self, raw_dataset, time_to_series):
        '''Convert a dataset to a different sample frequency, or another temporal conversion.'''
        raise NotImplementedError


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
