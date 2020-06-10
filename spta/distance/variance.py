'''
Analysis of variance of the distances: given a spatio-temporal region and a centroid,
calculate the standard deviation of the distances of each point to its centroid. Also,
plot a histogram of the distances to understand the nature of the variability.

Assumes that a distance matrix is available in the provided distance_measure instance.

TODO also support clusters: create as many subfigures as there are clusters, doing the same
analysis in each cluster and using a provided medoid as local centroid.
'''

import numpy as np
from matplotlib import pyplot as plt

from spta.util import log as log_util
from spta.util import maths as maths_util


def variance_analysis(spt_region, distance_measure, subplot=None, dist_max=None):
    '''
    Analysis of variance of the distances: given a spatio-temporal region and a centroid,
    calculate the standard deviation of the distances of each point to its centroid. Also,
    plot a histogram of the distances to understand the nature of the variability.

    Assumes that a distance matrix is available in the provided distance_measure instance.

    TODO also support clusters: create as many subfigures as there are clusters, doing the same
    analysis in each cluster and using a provided medoid as local centroid.
    '''

    # sanity checks, assumes required data is available instead of calculating it now
    assert spt_region.has_centroid()
    assert distance_measure.distance_matrix is not None

    # get all distances to centroid
    # see DistanceBetweenSeries.distances_to_point for details
    centroid = spt_region.get_centroid()
    all_point_indices = spt_region.all_point_indices
    distances_to_centroid = distance_measure.distances_to_point(spt_region, centroid,
                                                                all_point_indices)

    mean = np.mean(distances_to_centroid)
    sigma = np.std(distances_to_centroid)
    median = np.median(distances_to_centroid)

    # prepare a histogram of the distances
    # handle the case there is only a single histogram (whole region)
    fig = None
    as_subplot = True
    if not subplot:
        as_subplot = False
        fig, subplot = plt.subplots(1, 1)

    # plot the histogram
    # use 'doane' strategy for calculating bin size
    # maybe: use 'fd' strategy for calculating bin size (Freedman Diaconis Estimator)
    subplot.hist(distances_to_centroid, bins='doane')
    if not as_subplot:
        subplot.set_title('Distance Hist.')

    textstr = '\n'.join((
        '{}'.format(spt_region),
        'n={}'.format(len(all_point_indices)),
        r'$\mu=%.2f$' % (mean, ),
        r'$\sigma=%.2f$' % (sigma, ),
        r'M={:0.2f}'.format(median)))

    # place a text box in upper right in axes coords (0.05, 0.95 for left)
    subplot.text(0.75, 0.95, textstr, transform=subplot.transAxes, verticalalignment='top')
    subplot.set_xlabel('DTW distances')
    subplot.set_ylabel('Count')

    if dist_max:
        # for visual comparison in grid plot
        subplot.set_xlim((0, dist_max))

    if not as_subplot:
        plt.show()


def variance_analysis_clusters(clusters, distance_measure, plot_name=None):
    '''
    Calls variance_analysis for each cluster. Creates a subplot for each cluster, and organizes
    them in a single figure. Assumes that each cluster has its own centroid!
    '''
    logger = log_util.logger_for_me(variance_analysis_clusters)

    # create a suitable grid, use the first cluster shape to align the grid
    # this will fit for clusters based on square partitioning
    # set a size so that all subplots have predictable space
    _, x_len, y_len = clusters[0].shape
    div_x, div_y = maths_util.two_balanced_divisors_order_x_y(len(clusters), x_len, y_len)
    figsize = (div_y * 4.5, div_x * 4.5)

    logger.debug('Variance plot grid: ({}, {})'.format(div_x, div_y))
    logger.debug('Variance plot figsize: {}'.format(figsize))

    _, subplots = plt.subplots(div_x, div_y, figsize=figsize)

    # find the maximum distance from any point in a cluster to its centroid
    # this is done to ensure that all plots have the same x axis for visual comparison
    max_distance_within_cluster = -np.Inf

    for cluster in clusters:

        # first get all distances to the centroid
        cluster_centroid = cluster.get_centroid()
        cluster_point_indices = cluster.all_point_indices
        distances_to_centroid = distance_measure.distances_to_point(cluster, cluster_centroid,
                                                                    cluster_point_indices)

        # compute the maximum of this distances, and keep track of global maximum
        max_distance_to_centroid = np.max(distances_to_centroid)
        if max_distance_to_centroid > max_distance_within_cluster:
            max_distance_within_cluster = max_distance_to_centroid

    # call variance analysis on each cluster, use the grid subplots
    for i, cluster in enumerate(clusters):

        # if subplot is either (1, y) or (x, 1), it becomes one-dimensional, treat cases
        # don't bother with (1, 1), assume k >= 2
        index_x = int(i / div_y)
        index_y = i % div_y

        if div_x == 1:
            subplot = subplots[index_y]
        elif div_y == 1:
            subplot = subplots[index_x]
        else:
            subplot = subplots[index_x, index_y]

        variance_analysis(cluster, distance_measure, subplot=subplot,
                          dist_max=max_distance_within_cluster)

    # Save figure
    if plot_name:
        plt.draw()
        plt.savefig(plot_name)
        logger.info('Saved figure: {}'.format(plot_name))
    plt.show()


if __name__ == '__main__':
    log_util.setup_log('DEBUG')

    from spta.region import Point, Region
    from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata, \
        SpatioTemporalCluster
    from spta.region.centroid import CalculateCentroid
    from spta.region.mask import MaskRegionCrisp
    from spta.distance.dtw import DistanceByDTW

    # get a region with known centroid
    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)
    nordeste_small = SpatioTemporalRegion.from_metadata(nordeste_small_md)
    _, x_len, y_len = nordeste_small.shape
    nordeste_small.centroid = Point(4, 6)

    # load saved distances for this region
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_md(nordeste_small_md)

    # plot variance of entire region
    variance_analysis(nordeste_small, distance_measure, subplot=None)

    # similar but use regular partitioning
    k = 6
    clusters = []
    calculate_centroid = CalculateCentroid(distance_measure)

    for i in range(0, k):
        mask_i = MaskRegionCrisp.with_regular_partition(k, i, x_len, y_len)
        cluster_i = SpatioTemporalCluster(nordeste_small, mask_i)
        cluster_i.name = 'cluster{}'.format(i)

        # find centroid of the regular clusters
        # this also computes the distances to the centroid, so TODO use that
        cluster_i.centroid, _ = calculate_centroid.find_centroid_and_distances(cluster_i)
        clusters.append(cluster_i)

    # use cluster behavior to get a plot grid
    variance_analysis_clusters(clusters, distance_measure, plot_name='plots/test.pdf')
