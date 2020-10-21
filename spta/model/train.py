from spta.region import TimeInterval

from spta.util import log as log_util


class SplitTrainingAndTest(log_util.LoggerMixin):

    def __init__(self, test_len):
        super(SplitTrainingAndTest, self).__init__()
        self.test_len = test_len

    def split(self, spt_region):
        raise NotImplementedError


class SplitTrainingAndTestLast(SplitTrainingAndTest):
    '''
    Use the last entries elements in each time series, to split a spatio-temporal region into
    training region and test region.
    '''

    def split(self, spt_region):
        '''
        Given a spatio-temporal region, split the time series in two, to get a training region
        and a test region.

        spt_region
            spatio-temporal region with shape [(training_len + test_len), x_len, y_len]

        test_len
            Size of the test series that will be separated to create a test region. The rest of the
            series will form the training region.

        Output is a tuple of two spatio-temporal regions:
            training subset: shape [training_len, x_len, y_len]
            test_subset: shape [test_len, x_len, y_len]
        '''
        series_len = spt_region.series_len

        # use the concept of time intervals to easily split the region by the temporal axis
        training_size = series_len - self.test_len
        training_interval = TimeInterval(0, training_size)
        test_interval = TimeInterval(training_size, series_len)

        training_subset = spt_region.interval_subset(training_interval)
        test_subset = spt_region.interval_subset(test_interval)

        training_subset.name = 'train_{}'.format(spt_region)
        test_subset.name = 'test_{}'.format(spt_region)

        self.logger.debug('training_subset: {} -> {}'.format(training_subset,
                                                             training_subset.shape))
        self.logger.debug('test_subset: {} -> {}'.format(test_subset, test_subset.shape))

        return (training_subset, test_subset)
