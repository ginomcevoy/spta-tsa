'''
Example for k-medoids with synthetic data
'''
import logging
import numpy as np
import matplotlib.pyplot as plt
import random

from spta.dataset import synthetic_temporal as synth

from spta.region import Point
from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.region.distance import DistanceByDTW

from spta.util import plot as plot_util

from . import kmedoids
from . import get_medoid_indices


def alternating_functions_2d(spatial_region, series_len, function_options):
    '''
    Given a 2d region (x, y), create x*y series and form a spatio-temporal region.
    A function is chosen among function_options to create the series using
    synthetic.delayed_function_with_noise

    Just fill x*y/len(function_options) with the first, option, then for the second, etc.

    This is to test how k-medoids creates spatio-temporal clusters, expected outcome is that the
    clusters are identified and the plot colors are the same.
    '''

    (x_len, y_len) = spatial_region.shape

    # spt_numpy = np.zeros(x_len * y_len * series_len)
    # mask_numpy = np.zeros(x_len * y_len)
    spt_numpy = np.empty((series_len, x_len, y_len))
    mask_numpy = np.empty((x_len, y_len))

    options_n = len(function_options)

    random.seed(1)
    for i in range(0, x_len):

        for j in range(0, y_len):

            # given n function options,
            # create about 1/n of points_n with the first function, then 1/n for the second, etc
            # the decider is j (y-axis)

            # index = int(j * options_n / y_len)
            index = (i * j) % options_n
            # index = random.randint(0, options_n - 1)
            this_function = function_options[index]

            # create the data and add it to the matrix
            series = synth.delayed_function_with_noise(this_function, series_len)
            spt_numpy[:, i, j] = series
            mask_numpy[i, j] = index

    spt_region = SpatioTemporalRegion(spt_numpy)
    mask_region = SpatialRegion(mask_numpy)
    return (spt_region, mask_region)


def main():
    log_level = logging.DEBUG
    # log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()

    series_len = 200
    function_options = (synth.sine_function, synth.square_function,
                        synth.gaussian_function)

    # create the region to be clustered
    x_len, y_len = (8, 8)
    # region = Region(0, x_len, 0, y_len)
    # spatial_region = SpatialRegion.region_with_zeroes(region)
    region2d = np.empty((x_len, y_len))
    (spt_region, mask_region) = alternating_functions_2d(region2d, series_len, function_options)

    # plot the mask
    plot_util.plot_discrete_spatial_region(mask_region, 'Input mask')

    # 2d view of spatio temporal region, for plotting
    series_group = np.array(spt_region.as_list)
    plot_util.plot_series_group(series_group, series_len)

    # use DTW, pre-compute matrix
    distance_measure = DistanceByDTW()
    distance_measure.compute_distance_matrix(spt_region)

    # try these k, k=3 is best by design
    ks = [2, 3, 4]
    best_k, best_medoids, best_labels = kmedoids.silhouette_spt(ks, spt_region, distance_measure,
                                                                seed=1, max_iter=1000, tol=0.001,
                                                                verbose=True, show_graphs=True,
                                                                save_graphs=None)

    # Show best results
    logger.info('Best k: {}'.format(best_k))
    logger.info('Best medoids: {}'.format(str(get_medoid_indices(best_medoids))))
    logger.info('Best labels: {}'.format(str(best_labels)))
    plot_util.plot_series_group_by_color(series_group, series_len, best_labels)


if __name__ == '__main__':
    main()
