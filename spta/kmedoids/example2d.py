'''
Example for k-medoids with synthetic data
'''
import numpy as np
import random

from spta.dataset import synthetic_temporal as synth

from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.distance.dtw import DistanceByDTW

from spta.util import plot as plot_util
from spta.util import log as log_util

from . import silhouette
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
    logger = log_util.setup_log('DEBUG')

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
    random_seeds = (1,)

    metadata = silhouette.silhouette_default_metadata(ks, random_seeds, distance_measure)
    best_k, _, best_medoids, best_membership = silhouette.do_silhouette_analysis(spt_region,
                                                                                 metadata)

    # Show best results
    logger.info('Best k: {}'.format(best_k))
    logger.info('Best medoids: {}'.format(str(get_medoid_indices(best_medoids))))
    logger.info('Best membership: {}'.format(str(best_membership)))
    plot_util.plot_series_group_by_color(series_group, series_len, best_membership)


if __name__ == '__main__':
    main()
