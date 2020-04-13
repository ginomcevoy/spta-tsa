import logging
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


PALETTE = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot_series_group(series_group, series_len):
    '''
    Plots a group of series all in the same graph, assumes all series have same length.
    '''

    plt.figure()

    x = range(0, series_len)
    for series in series_group:
        plt.plot(x, series)

    plt.show()


def plot_series_group_by_color(series_group, series_len, colors):
    '''
    Plots a group of series all in the same graph, assumes all series have same length.
    Uses the provided color index for the palette, should be the same length as the series_group
    '''

    check_palette_len(len(set(colors)))

    plt.figure()

    x = range(0, series_len)
    for (index, series) in enumerate(series_group):
        plt.plot(x, series, c=PALETTE[colors[index]])

    plt.show()


def plot_discrete_spatial_region(spatial_region, title='', subplot=None):
    '''
    Plots a discrete spatial region with a scatter plot, where the values are assigned to colors.
    '''

    if not subplot:
        _, subplot = plt.subplots(1, 1)

    region_np_array = spatial_region.as_numpy
    # plt.imshow(region_np_array, cmap='hot', interpolation='nearest')
    subplot.imshow(region_np_array, interpolation='nearest')

    if title:
        subplot.set_title(title)

    if not subplot:
        plt.show()


def plot_clustering_silhouette(distance_matrix, cluster_labels, subplot=None):
    '''
    Plot a silhouette to graphically display a measure of the efficiency of a clustering
    algorithm, e.g. k-medoids. Assumes that the distance matrix has already been calculated.

    Based on:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    '''
    logger = logging.getLogger()
    logger.debug('Shape of distance_matrix: {}'.format(str(distance_matrix.shape)))
    logger.debug('len(distance_matrix): {}'.format(str(len(distance_matrix))))

    if not subplot:
        _, subplot = plt.subplots(1, 1)

    # infer the value of k (number of clusters)
    k = len(set(cluster_labels))

    # The silhouette coefficient can range from -1, 1, in practice it may be -0.1 to 1
    subplot.set_xlim([-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    subplot.set_ylim([0, len(distance_matrix) + (k + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters.
    # Since we are using DTW for distance, we need to precalculate the distances (UGH)
    silhouette_avg = silhouette_score(X=distance_matrix, labels=cluster_labels,
                                      metric="precomputed")

    logger.info('The average silhouette_score for k={} is: {}'.format(k, silhouette_avg))

    # Compute the silhouette scores for each sample
    # Since we are using DTW for distance, we need to precalculate the distances (UGH)
    sample_silhouette_values = silhouette_samples(distance_matrix, cluster_labels,
                                                  metric="precomputed")

    y_lower = 10
    for i in range(0, k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        subplot.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        subplot.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    subplot.set_title("The silhouette plot for the various clusters.")
    subplot.set_xlabel("The silhouette coefficient values")
    subplot.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    subplot.axvline(x=silhouette_avg, color="red", linestyle="--")

    subplot.set_yticks([])  # Clear the yaxis labels / ticks
    subplot.set_xticks(np.linspace(-1, 1, 9))

    if not subplot:
        plt.show()

    # return the metric for analysis
    return silhouette_avg


def check_palette_len(color_length):
    if color_length > len(PALETTE):
        raise ValueError('Need more colors in palette!')


if __name__ == '__main__':

    from spta.region.spatial import SpatialRegion

    x_len = 9
    y_len = 8

    data = (0, 1, 2)
    repeats = int((x_len * y_len) / len(data))
    list_1d = np.repeat(data, repeats)  # [0, 0, 0, ..., 1, 1, 1, ... 2, 2, ... 2]

    # switch it up a bit
    list_1d[repeats: repeats + 2] = 0
    list_1d[-repeats - 2: -repeats] = 2
    print(list_1d)

    region = SpatialRegion.create_from_1d(list_1d, x_len, y_len)
    print(region)

    plot_discrete_spatial_region(region)
