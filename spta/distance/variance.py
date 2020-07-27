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


class DistanceHistogram(log_util.LoggerMixin):
    '''
    Creates a histogram of the distance of a list of points to another specified point.

    The distance_measure.distances_to_point() method is used to access the distances between a
    list of points via their indices, and another point specified as Point(i, j).

    The point_indices property method computes the required indices for
    distance_measure.distances_to_point(). Here, we access all indices in the specified
    spatio-temporal region.

    distances_to_point(point)

    The distances_to_point() method finds the distances. Will also save the maximum value of these
    distances, useful when plotting.

    The histogram() method will call subplot.hist() on a provided subplot, if available. If subplot
    is not available, creates a full figure and calls hist() on it. Will also calculate some
    statistics about the distances and add them to the plot.
    '''
    def __init__(self, spt_region, distance_measure, bins='auto'):
        '''
        If bins is a string, it is one of the binning strategies supported by
        numpy.histogram_bin_edges:
        'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        '''
        # convenience check, assumes required data is available instead of calculating it now
        assert distance_measure.distance_matrix is not None

        self.spt_region = spt_region
        self.distance_measure = distance_measure
        self.bins = bins

    @property
    def point_indices(self):
        '''
        Get indices relevant to this region. Here, we get all indices. Subclasses may have
        more interesting behavior.
        '''
        return self.spt_region.all_point_indices

    def distances_to_point(self, point_of_interest):
        '''
        Compute the distances to the point of interest, e.g. centroid.
        '''
        return self.distance_measure.distances_to_point(self.spt_region, point_of_interest,
                                                        self.point_indices)

    def max_distance(self, distances, max_override=None):
        # idiom to use max_override only if provided
        if max_override is None:
            max_override = -np.inf

        # overall maximum value
        return np.max([np.max(distances), max_override])

    def statistics(self, distances):
        '''
        Caculates some statistics about the distances, and returns them as text.
        '''

        # min is the minimum distance after removing the mandatory 0 value from the point of
        # interest, this idiom puts the second-lowest distance at the right place
        # https://stackoverflow.com/a/22546769
        min_distance_not_zero = np.partition(distances, 1)[1]
        max_distance = np.max(distances)

        mean = np.mean(distances)
        sigma = np.std(distances)
        median = np.median(distances)

        textstr = '\n'.join((
            '{}'.format(self.spt_region),
            'n = {}'.format(len(self.point_indices)),
            'min = {:.2f}'.format(min_distance_not_zero),
            'max =  {:.2f}'.format(max_distance),
            'x̅ = {:.2f}'.format(mean),
            's = {:.2f}'.format(sigma),
            'M = {:0.2f}'.format(median)))

        return textstr

    def plot(self, distances, subplot=False, with_statistics=True, max_override=None,
             plot_name=None, alpha=1):
        '''
        Plot the histogram of distances to the point.

        distances
            A result from distances_to_point

        subplot
            Is this part of a bigger figure? If so, then a proper subplot axis is expected.

        with_statistics
            Add statistics to the plot iff True

        max_override
            Optional maximum distance for x axis.

        plot_name
            If provided, save the plot with this name.

        alpha
            Value [0, 1] for alpha channel transparency.
            Useful when adding more histograms to the plot. Default value of 1 disables
            transparency.
        '''

        # prepare a histogram of the distances
        # handle the case there is only a single histogram (whole region)
        fig = None
        as_subplot = True
        if not subplot:
            as_subplot = False
            fig, subplot = plt.subplots(1, 1)

        # logging only
        mean_cost = np.mean(distances)
        np.set_printoptions(precision=3)
        # self.logger.debug('Distances: {}'.format(distances))
        self.logger.debug('Mean from histogram = {:.3f}'.format(mean_cost))

        # plot the histogram
        subplot.hist(distances, label=str(self.spt_region), bins=self.bins, alpha=alpha)
        if not as_subplot:
            subplot.set_title('Distance Hist.')

        if max_override:
            # for visual comparison in grid plot
            subplot.set_xlim((0, max_override))

        subplot.set_xlabel('DTW distances')
        subplot.set_ylabel('Count')

        if with_statistics:
            # place a text box in upper right in axes coords (0.05, 0.95 for left)
            textstr = self.statistics(distances)
            subplot.text(0.70, 0.95, textstr, transform=subplot.transAxes,
                         verticalalignment='top', bbox=dict(edgecolor='black', fill=False))

        # Save figure
        if plot_name and not as_subplot:
            plt.draw()
            plt.savefig(plot_name)
            self.logger.info('Saved figure: {}'.format(plot_name))

        if not as_subplot:
            plt.show()

        return subplot

    @classmethod
    def histogram_to_centroid(cls, spt_region, distance_measure, bins='auto', plot_name=None):
        '''
        Plot the histogram of distances to the centroid of the specified region.
        Assumes that the centroid has been pre-calculated.
        '''

        # sanity checks, assumes required data is available instead of calculating it now
        assert spt_region.has_centroid()
        assert distance_measure.distance_matrix is not None

        distance_hist = DistanceHistogram(spt_region, distance_measure, bins)
        distances = distance_hist.distances_to_point(spt_region.get_centroid())
        max_override = distance_hist.max_distance(distances)
        distance_hist.plot(distances,
                           with_statistics=True,
                           subplot=False,
                           max_override=max_override,
                           plot_name=plot_name)

        return distance_hist


class DistanceHistogramWithRandomPointsOtherClusters(DistanceHistogram):
    '''
    Given a distance histogram of a spatio-temporal cluster, adds a histogram of random points
    from other clusters on top of it, using the decorator pattern.
    Assumes that the provided spatio-temporal region is a subset of the whole region given by
    its (x_len, y_len) coordinates. Also assumes that a subplot is provided for adding the hist.

    See point_indices for details on calculating the random points.
    '''

    def __init__(self, cluster_histogram, num_points=0):
        self.cluster_histogram = cluster_histogram
        self.num_points = num_points

        self.spt_region = cluster_histogram.spt_region
        self.distance_measure = cluster_histogram.distance_measure

        # assuming a cluster
        self.spt_cluster = self.spt_region

    @property
    def point_indices(self):
        '''
        Here, random indices outside of the cluster are calculated.
        The random points are calculated using the x_len, y_len region of the cluster, and calling
        spta.util.maths.evaluate_random_array_of_integers to get points in (x_len, y_len) Region
        that do not belong to the points in the current cluster.

        Will fail if there are no points outside of the cluster!
        '''
        indices_of_cluster = self.spt_cluster.all_point_indices

        _, x_len, y_len = self.spt_cluster.shape
        num_remaining_points_in_region = x_len * y_len - len(indices_of_cluster)

        # requested number of random points, same number of points in cluster by default
        if self.num_points == 0:
            self.num_points = len(indices_of_cluster)

        # fall back to available points if there are not enough to meet request
        if self.num_points > num_remaining_points_in_region:
            self.num_points = num_remaining_points_in_region

        # Find random indices outside of cluster.
        # If request is too high, then it finds all points outside the cluster.
        return maths_util.random_integers_with_blacklist(n=self.num_points,
                                                         min_value=0,
                                                         max_value=(x_len * y_len - 1),
                                                         blacklist=indices_of_cluster)

    def distances_to_point(self, point_of_interest):
        '''
        Compute the distances to the point of interest, e.g. centroid.
        Here we decorate original behavior to compute the original histogram distances *and* the
        random distances.
        '''
        cluster_distances = self.cluster_histogram.distances_to_point(point_of_interest)

        # leverage original behavior
        parent = super(DistanceHistogramWithRandomPointsOtherClusters, self)
        random_distances = parent.distances_to_point(point_of_interest)
        return (cluster_distances, random_distances)

    def max_distance(self, distances, max_override=None):
        '''
        Calculate the max distance considering cluster and random distances
        '''
        # idiom to use max_override only if provided
        if max_override is None:
            max_override = -np.inf

        cluster_distances, random_distances = distances

        # overall maximum value
        return np.max([np.max(cluster_distances), np.max(random_distances), max_override])

    def statistics(self, distances):
        '''
        Decorates the statistics
        '''

        # get the normal statistics
        cluster_distances, random_distances = distances
        textstr_cluster = self.cluster_histogram.statistics(cluster_distances)

        # add statistics from random points: number of points, number of points that 'intersect'
        # the original histogram
        # the interesection is calculated by counting how many random_distances are lower than the
        # maximum cluster distance
        smaller_random_distances = np.where(random_distances < np.max(cluster_distances))[0]
        textstr = '\n'.join((
            textstr_cluster,
            '#rand = {}'.format(len(random_distances)),
            '∩rand = {}'.format(len(smaller_random_distances)),
        ))

        return textstr

    def plot(self, distances, subplot=False, with_statistics=True, max_override=None,
             plot_name=None, alpha=1):

        # only valid within an existing subplot
        assert subplot is not None

        # assume distances_to_point was called... ugly yes
        cluster_distances, random_distances = distances

        # logging only
        mean_cost_random = np.mean(random_distances)
        np.set_printoptions(precision=3)
        self.logger.debug('Cost from random points histogram = {:.3f}'.format(mean_cost_random))

        # plot the original histogram in the subplot
        # does not save this plot, does not add statistics
        self.cluster_histogram.plot(distances=cluster_distances,
                                    subplot=subplot,
                                    with_statistics=False,
                                    plot_name=None,
                                    alpha=alpha)

        # add the histogram of provided distances
        # useful for adding a second histogram to existing plot
        subplot.hist(random_distances, label='random', bins=self.cluster_histogram.bins,
                     alpha=alpha)

        if with_statistics:
            # use the decorated statistics
            # place a text box in upper right in axes coords (0.05, 0.95 for left)
            textstr = self.statistics(distances)
            subplot.text(0.70, 0.95, textstr, transform=subplot.transAxes,
                         verticalalignment='top', bbox=dict(edgecolor='black', fill=False))

        if max_override:
            # for visual comparison in grid plot
            subplot.set_xlim((0, max_override))

        subplot.legend(loc='upper left')

        # Save figure
        if plot_name:
            plt.draw()
            plt.savefig(plot_name)
            self.logger.info('Saved figure: {}'.format(plot_name))

        return subplot


class DistanceHistogramRandomOutsidePoints(DistanceHistogram):
    '''
    Given a spatio-temporal cluster, adds a histogram of random points from other clusters.
    Assumes that the provided spatio-temporal region is a subset of the whole region given by
    its (x_len, y_len) coordinates. Also assumes that a subplot is provided for adding the hist.

    See point_indices for details on calculating the random points.
    '''

    def __init__(self, spt_cluster, distance_measure, bins='auto', num_points=0):
        '''
        Also pass an optional num_points parameter to indicate how many random points are
        used. If not provided, then use either:
            - As many random points as there are in the cluster
            - All available points, if this number is smaller than the points in the cluster
        '''
        super(DistanceHistogramRandomOutsidePoints, self).__init__(spt_cluster, distance_measure,
                                                                   bins)
        self.num_points = num_points

        # alias for spt_region
        self.spt_cluster = spt_cluster

    @property
    def point_indices(self):
        '''
        Here, random indices outside of the cluster are calculated.
        The random points are calculated using the x_len, y_len region of the cluster, and calling
        spta.util.maths.evaluate_random_array_of_integers to get points in (x_len, y_len) Region
        that do not belong to the points in the current cluster.

        Will fail if there are no points outside of the cluster!
        '''
        indices_of_cluster = self.spt_cluster.all_point_indices

        _, x_len, y_len = self.spt_cluster.shape
        num_remaining_points_in_region = x_len * y_len - len(indices_of_cluster)

        # requested number of random points, same number of points in cluster by default
        if self.num_points == 0:
            self.num_points = len(indices_of_cluster)

        # fall back to available points if there are not enough to meet request
        if self.num_points > num_remaining_points_in_region:
            self.num_points = num_remaining_points_in_region

        # Find random indices outside of cluster.
        # If request is too high, then it finds all points outside the cluster.
        return maths_util.random_integers_with_blacklist(n=self.num_points,
                                                         min_value=0,
                                                         max_value=(x_len * y_len - 1),
                                                         blacklist=indices_of_cluster)

    def plot(self, distances, subplot, alpha=0.5, max_override=None):

        # only valid within an existing subplot
        assert subplot is not None

        # logging only
        mean_cost = np.mean(distances)
        np.set_printoptions(precision=3)
        # self.logger.debug('Distances random points: {}'.format(distances))
        self.logger.debug('Cost from random points histogram = {:.3f}'.format(mean_cost))

        # add the histogram of provided distances
        # useful for adding a second histogram to existing plot
        subplot.hist(distances, label='random', bins=self.bins, alpha=alpha)

        if max_override:
            # for visual comparison in grid plot
            subplot.set_xlim((0, max_override))

        subplot.legend(loc='upper left')

        return subplot


class DistanceHistogramClusters(DistanceHistogram):
    '''
    Given a cluster partition, plot one distance histogram per cluster, and show all subplots
    in a single figure. Uses the Composite pattern to plot a matrix of subplots, also all
    distances are relative to each cluster centroid.

    For clusters, optionally show random points outside each cluster.

    The plots are arranged so that they match the position when regular partitioning is used.
    See spta.util.maths.two_balanced_divisors_order_x_y for details.
    '''

    def __init__(self, clusters, clustering_algorithm, random_points=None, bins='auto'):
        '''
        clusters
            a list of SpatioTemporalCluster instances

        clustering_algorithm
            a clustering algorithm, used to extract metadata

        random_points
            Indicates how many random points will be added from other clusters to each hist. plot.
            If 0, add the same number of points that the cluster has.
            If None, do not add random points.
            See DistanceHistogramRandomOutsidePoints for details.

        bins
            How many bins to use for histogram.
        '''
        # convenience checks, assumes required data is available instead of calculating it now
        for cluster in clusters:
            assert cluster.has_centroid()

        self.clusters = clusters
        self.clustering_algorithm = clustering_algorithm
        self.random_points = random_points
        self.bins = bins

        distance_measure = clustering_algorithm.distance_measure

        # composite pattern: this histogram plot has k histogram subplots inside it.
        self.histograms = []
        for cluster in clusters:

            # histogram for the points in a cluster
            histogram = DistanceHistogram(cluster, distance_measure, bins)

            if random_points is not None:
                # also add the histogram of random points on top of the clustering,
                # done by decorating the original histogram
                histogram = DistanceHistogramWithRandomPointsOtherClusters(histogram,
                                                                           random_points)
            self.histograms.append(histogram)

        # saved as a flag for plot name
        self.random_points = random_points

        # self.histograms = [
        #     DistanceHistogram(cluster, distance_measure, bins)
        #     for cluster
        #     in clusters
        # ]

        # # if random_points is provided, then also add their histograms to existing subplots.
        # self.random_histograms = None
        # if random_points is not None:
            # self.random_histograms = [
            #     DistanceHistogramRandomOutsidePoints(cluster, distance_measure, bins=bins,
            #                                          num_points=random_points)
            #     for cluster
            #     in clusters
            # ]

    @property
    def point_indices(self):
        raise NotImplementedError

    def distances_to_point(self, point_of_interest):
        raise NotImplementedError

    def max_distance(self):
        '''
        Calculate maximum overall distance among all cluster distances.
        Uses the centroids of each cluster to find the distances for each cluster.
        '''
        # idiom to find overall maximum
        max_dist = -np.Inf

        # iterate clusters and access all histograms
        for i, cluster_i in enumerate(self.clusters):
            cluster_i_hist = self.histograms[i]

            # find the distances to the centroid
            # should also work with decorated histograms
            cluster_i_dists = cluster_i_hist.distances_to_point(cluster_i.centroid)

            # since max_distance keeps maximum, use this instead of asking which is larger
            # should also work with decorated histograms
            max_dist = cluster_i_hist.max_distance(cluster_i_dists, max_dist)

            # # also evaluate random histograms if available
            # if self.random_histograms:
            #     random_i_hist = self.random_histograms[i]
            #     random_i_dists = random_i_hist.distances_to_point(cluster_i.centroid)
            #     max_dist = random_i_hist.max_distance(random_i_dists, max_dist)

        # at the end of the loop, the maximum distance between each point and its centroid is found
        return max_dist

    def statistics(self):
        raise NotImplementedError

    def plot(self, with_statistics=True, plot_dir=None, alpha=0.5):
        '''
        Composes a plot with subplots, one or each cluster.
        '''

        # create a suitable grid, use the first cluster shape to align the grid
        # this will fit for clusters based on regular partitioning
        # set a size so that all subplots have predictable space
        _, x_len, y_len = self.clusters[0].shape
        div_x, div_y = maths_util.two_balanced_divisors_order_x_y(len(self.clusters), x_len, y_len)
        figsize = (div_y * 4.5, div_x * 4.5)

        self.logger.debug('Variance plot grid: ({}, {})'.format(div_x, div_y))
        self.logger.debug('Variance plot figsize: {}'.format(figsize))

        _, subplots = plt.subplots(div_x, div_y, figsize=figsize)

        # use maximum distance to override x_lim of all subplots
        max_override = self.max_distance()

        # if saving plot, the filename is calculated using the clustering algorithm and possible
        # random points
        plot_name = None
        if plot_dir is not None:
            plot_partial_name = 'variance-histograms__{!r}'.format(self.clustering_algorithm)

            if self.random_points is not None:
                # if random points are provided, change the filename to avoid overwrite
                plot_partial_name = plot_partial_name + '__random{}'.format(self.random_points)

            plot_name = '{}/{}.pdf'.format(plot_dir, plot_partial_name)

        # plot histogram of each cluster, use the grid subplots
        for i, cluster in enumerate(self.clusters):

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

            # access each histogram and compose
            # don't pass plot_name to avoid saving each subplot
            # should also work with decorated histograms
            histogram_i = self.histograms[i]
            distances_i = histogram_i.distances_to_point(cluster.centroid)
            histogram_i.plot(distances=distances_i,
                             subplot=subplot,
                             with_statistics=with_statistics,
                             plot_name=None,
                             alpha=alpha,
                             max_override=max_override)

            # # if there are random histograms, add them to the corresponding subplot
            # if self.random_histograms:
            #     random_hist_i = self.random_histograms[i]
            #     random_distances_i = random_hist_i.distances_to_point(cluster.centroid)

            #     # this will add the random histogram to the histogram of this cluster
            #     # also override x_lim for visual comparison
            #     random_hist_i.plot(distances=random_distances_i,
            #                        subplot=subplot,
            #                        alpha=alpha,
            #                        max_override=max_override)
        # Save figure
        if plot_name:
            plt.draw()
            plt.savefig(plot_name)
            self.logger.info('Saved figure: {}'.format(plot_name))
        plt.show()

    @classmethod
    def cluster_histograms(cls, clusters, clustering_algorithm, random_points=None, bins='auto',
                           with_statistics=True, plot_dir=None, alpha=0.5):
        '''
        Create a figure with histogram subplots for each cluster, with distances relative
        to the centroid of each cluster.
        '''
        cluster_hist = DistanceHistogramClusters(clusters, clustering_algorithm, random_points,
                                                 bins)
        cluster_hist.plot(with_statistics=with_statistics,
                          plot_dir=plot_dir,
                          alpha=alpha)


if __name__ == '__main__':
    log_util.setup_log('DEBUG')

    from experiments.metadata.region import predefined_regions
    from spta.clustering.regular import RegularClusteringAlgorithm, RegularClusteringMetadata
    from spta.region.centroid import CalculateCentroid
    from spta.distance.dtw import DistanceByDTW

    spt_region_md = predefined_regions()['sp_small']
    spt_region = spt_region_md.create_instance()
    _, x_len, y_len = spt_region.shape

    # load saved distances for this region
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_md(spt_region_md)

    # find centroid of region
    calculate_centroid = CalculateCentroid(distance_measure)
    spt_region.centroid, _ = calculate_centroid.find_centroid_and_distances(spt_region)

    # plot variance of entire region
    DistanceHistogram.histogram_to_centroid(spt_region, distance_measure,
                                            plot_name='plots/variance_test1.pdf', bins='auto')

    # use regular partitioning for clusters
    regular_clustering = RegularClusteringAlgorithm(RegularClusteringMetadata(k=4),
                                                    distance_measure)
    partition = regular_clustering.partition(spt_region, with_medoids=True)
    clusters = partition.create_all_spt_clusters(spt_region, medoids=partition.medoids)

    DistanceHistogramClusters.cluster_histograms(clusters=clusters,
                                                 clustering_algorithm=regular_clustering,
                                                 random_points=None,
                                                 bins='auto',
                                                 with_statistics=True,
                                                 plot_dir='plots',
                                                 alpha=0.5)

    DistanceHistogramClusters.cluster_histograms(clusters=clusters,
                                                 clustering_algorithm=regular_clustering,
                                                 random_points=0,
                                                 bins='auto',
                                                 with_statistics=True,
                                                 plot_dir='plots',
                                                 alpha=0.5)
