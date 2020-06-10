'''
Example for k-medoids with synthetic data
'''
import numpy as np

from spta.dataset import synthetic_temporal as synth
from spta.distance.dtw import DistanceByDTW
from spta.util import plot as plot_util
from spta.util import log as log_util

from . import kmedoids


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
    logger = log_util.setup_log('DEBUG')

    series_n = 18
    series_len = 200
    function_options = (synth.sine_function, synth.square_function,
                        synth.gaussian_function)

    precompute_distance_matrix = True

    # create the sequence group to be clustered, plot it
    series_group = alternating_functions_1d(series_n, series_len, function_options)
    plot_util.plot_series_group(series_group, series_len)

    distance_measure = DistanceByDTW()
    if precompute_distance_matrix:

        # pre-calculate all distances using DTW
        distance_measure.compute_distance_matrix(series_group)

    # apply k-medoids on the data using DTW
    k = len(function_options)
    kmedoids_result = kmedoids.run_kmedoids(series_group, k, distance_measure, random_seed=1,
                                            max_iter=1000, tol=0.001, verbose=True)

    logger.info('Medoids: {}'.format(str(kmedoids_result.medoids)))
    logger.info('Labels: {}'.format(str(kmedoids_result.labels)))
    plot_util.plot_series_group_by_color(series_group, series_len, kmedoids_result.labels)


if __name__ == '__main__':
    main()
