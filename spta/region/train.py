from spta.region import TimeInterval
from spta.util import log as log_util

'''
How to extract test values from a series.
'''
TEST_EXTRACTION_METHODS = ('last',)


def split_region_in_train_test(spt_region, test_len, test_from='last'):
    '''
    Given a spatio-temporal region, split the time series in two, to get a training region
    and a test region.

    spt_region
        spatio-temporal region with shape [(training_len + test_len), x_len, y_len]

    test_len
        Size of the test series that will be separated to create a test region. The rest of the
        series will form the training region.

    test_from
        A string that should match one of TEST_EXTRACTION_METHODS.
        'last': use the last elements in the input region as test series.

    Output is a tuple of two spatio-temporal regions:
        training subset: shape [training_len, x_len, y_len]
        test_subset: shape [test_len, x_len, y_len]
    '''

    series_len = spt_region.series_len()

    # TODO: handle other extraction methods, assume 'last' is used
    # divide series in training and test, training come first in the series
    assert test_from == 'last'

    # use the concept of time intervals to easily split the region by the temporal axis
    training_size = series_len - test_len
    training_interval = TimeInterval(0, training_size)
    test_interval = TimeInterval(training_size, series_len)

    training_subset = spt_region.interval_subset(training_interval)
    test_subset = spt_region.interval_subset(test_interval)

    logger = log_util.logger_for_me(split_region_in_train_test)
    logger.debug('training_subset: {} {}'.format(training_subset.__class__.__name__,
                                                 training_subset.shape))
    logger.debug('test_subset: {} {}'.format(test_subset.__class__.__name__,
                                             test_subset.shape))

    return (training_subset, test_subset)
