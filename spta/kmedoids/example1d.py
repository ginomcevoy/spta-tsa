'''
Example for k-medoids with synthetic data
'''
import logging
import numpy as np

from spta.dataset import synthetic_temporal as synth
from spta.region.distance import DistanceByDTW
from spta.util import plot as plot_util

from . import kmedoids
from . import get_medoid_indices


def alternating_functions_1d(series_n, series_len, function_options):
    '''
    Create n series, forming a group of series (series_group).
    A function is chosen among function_options to create the series using
    synthetic.delayed_function_with_noise

    Just alternate sequentially for now.

    This is to test how k-medoids groups the series, expected outcome is that the sequence is
    identified.
    '''
    series_group = list()
    for i in range(0, series_n):
        # since we draw sequentially, use the i index
        index = i % len(function_options)
        this_function = function_options[index]

        # create the data and add to group
        series = synth.delayed_function_with_noise(this_function, series_len)
        series_group.append(series)

    series_group = np.array(series_group)
    return series_group


def main():
    # log_level = logging.DEBUG
    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()

    series_n = 18
    series_len = 200
    function_options = (synth.sine_function, synth.square_function,
                        synth.gaussian_function)

    # create the sequence group to be clustered, plot it
    series_group = alternating_functions_1d(series_n, series_len, function_options)
    plot_util.plot_series_group(series_group, series_len)

    # apply k-medoids on the data using DTW
    k = len(function_options)
    distance_measure = DistanceByDTW()
    kmedoids_result = kmedoids.run_kmedoids(series_group, k, distance_measure, seed=1,
                                            max_iter=1000, tol=0.001, verbose=True)
    (medoids, labels, costs, _, _) = kmedoids_result

    logger.info('Medoids: {}'.format(str(get_medoid_indices(medoids))))
    logger.info('Labels: {}'.format(str(labels)))
    plot_util.plot_series_group_by_color(series_group, series_len, labels)


if __name__ == '__main__':
    main()
