import logging
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import seaborn as sns

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


def plot_discrete_spatial_region(spatial_region, title='', clusters=True, subplot=None,
                                 mark_points=None):
    '''
    Plots a discrete spatial region with a scatter plot, where the values are assigned to colors.
    '''

    fig = None
    had_subplot = True
    if not subplot:
        fig, subplot = plt.subplots(1, 1)
        had_subplot = False

    region_np_array = spatial_region.as_numpy

    if clusters:
        # assume that we want to plot cluster labels, which are between 0 and k.
        # the idea is to match silhouette colors, so we will use cm.nipy_spectral function

        # imshow receives (M, N), (M, N, 3) or (M, N, 4) array
        # we want to pass (M, N, 4) array, where the values in (4) are obtained using nipy_spectral

        # get the label values between 0 and 1
        k = np.max(region_np_array) + 1
        label_np_array = region_np_array * 1.0 / k

        # apply cm.nipy_spectral over each point in the region
        # this returns a list of arrays
        label_colors_list = list(map(cm.nipy_spectral, label_np_array))

        # concatenate the lists into a single array, recover region shape and add the 4 RGBA
        # FIXME COORDS this is rotated... but then again our coordinates are *also* rotated
        color_shape = (region_np_array.shape[0], region_np_array.shape[1], 4)
        color_region = np.concatenate(label_colors_list, axis=0).reshape(color_shape)

        # now call imshow with our (x_len, y_len, 4) color data
        imgplot = subplot.imshow(color_region)
        imgplot.set_cmap('nipy_spectral')

        if mark_points is not None:
            for point in mark_points:

                point_pos = (point.x, point.y)

                # add points as black 'x'...
                point_symbol = 'xk'

                # except for when the color is black, then we want to use white
                # this idiom compares two arrays
                if (color_region[point_pos] == np.array([0., 0., 0., 1.])).all():
                    point_symbol = 'xw'

                # FIXME COORDS see plot_partition on details of inversion
                subplot.plot(point_pos[1], point_pos[0], point_symbol)

    else:
        # this does produce the colored heat map but colors won't match with a silhouette plot
        subplot.imshow(region_np_array, interpolation='nearest')

    if title:
        subplot.set_title(title)

    if not had_subplot:
        plt.show()

    return fig


def plot_2d_clusters(cluster_labels, shape_2d, title='', subplot=None):
    '''
    Convenience function for plotting cluster labels in a 2d region, effectively displaying the
    partitioning of a region.
    '''
    x_len, y_len = shape_2d

    # import here to avoid any circular imports
    from spta.region.spatial import SpatialRegion
    label_region = SpatialRegion.create_from_1d(cluster_labels, x_len, y_len)
    return plot_discrete_spatial_region(label_region, title, True, subplot)


def plot_partition(partition, title='', subplot=None, rectangle_region=None, mark_points=None):
    '''
    Convenience function for plotting a clustering partition.

    If rectangle_region is provided, it will also draw a red rectangle representing a region of
    interest within the partition plot. Useful for prediction regions in solvers.
    '''
    fig = plot_discrete_spatial_region(partition, title, True, subplot, mark_points)

    plot_here = fig
    if subplot is not None:
        # if subplot is provided, then we draw in the subplot
        plot_here = subplot

    if rectangle_region is not None:

        # FIXME COORDS The plot in plot_2d_clusters is "wrong", meaning that the axes are rotated.
        # However, the x, y coordinates are *also* rotated (x is latitute, should be longitude).
        # So the plot is "OK", but the rectangle region needs to be rotated here.

        # When the (x, y) coordinates are changed to long/lat (instead of lat/long), then we
        # can come back here again and revert this rotation.
        xy = (rectangle_region.y1 - 0.5, rectangle_region.x1 - 0.5)
        width = rectangle_region.y2 - rectangle_region.y1
        height = rectangle_region.x2 - rectangle_region.x1

        # https://stackoverflow.com/questions/52056475/python-plot-rectangles-of-known-size-at-scatter-points
        plot_here.add_patch(Rectangle(xy=xy, width=width, height=height, linewidth=1,
                                      color='red', fill=False))

    return fig


def plot_clustering_silhouette(distance_matrix, cluster_labels, subplot=None, show_graphs=True):
    '''
    Plot a silhouette to graphically display a measure of the efficiency of a clustering
    algorithm, e.g. k-medoids. Assumes that the distance matrix has already been calculated.

    Based on:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    # TODO move some of this to kmedoids.silhouette?
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

    subplot.set_title('Silhouette plot, score={:1.3f}'.format(silhouette_avg))
    subplot.set_xlabel("The silhouette coefficient values")
    subplot.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    subplot.axvline(x=silhouette_avg, color="red", linestyle="--")

    subplot.set_yticks([])  # Clear the yaxis labels / ticks
    subplot.set_xticks(np.linspace(-1, 1, 9))

    if not subplot and show_graphs:
        plt.show()

    # return the metric for analysis
    return silhouette_avg


def plot_distances_vs_forecast_errors(distances_to_point, forecast_errors, distance_measure,
                                      error_type, info_text, plot_filename=None,
                                      title='Distances to medoid vs forecast errors at medoid'):

    logger = logging.getLogger()
    _, subplot = plt.subplots(1, 1, figsize=(7, 5))
    subplot.plot(distances_to_point, forecast_errors, 'bo')

    subplot.set_title(title)

    # add some info about plot
    if info_text:
        subplot.text(0.05, 0.95, info_text, transform=subplot.transAxes,
                     verticalalignment='top')

    subplot.set_xlabel('Distances to medoid ({})'.format(distance_measure))
    subplot.set_ylabel('{} forecast errors'.format(error_type))
    subplot.grid(True, linestyle='--')
    if plot_filename:
        plt.draw()
        plt.savefig(plot_filename)
        logger.info('Saved figure: {}'.format(plot_filename))
    plt.show()


def plot_pandas_heatmap(df, x_col, y_col, value_col, scale=(0, 1), ax=None):
    '''
    Given a pandas DataFrame object representing values over a 2D region, build a heatmap that
    conserves the shape of the region. This assumes that the region has been tiled in rectangles
    of the same shape, so that each tuple of the dataframe contains the  (x, y) coordinates of a corner
    of the rectangle for which the value applies.

    df
        pandas DataFrame

    x_col
        column name for the x coordinates of the tiles

    y_col
        column name for the y coordinates of the tiles

    value_col
        column name for the value at (x, y)

    x_width
        width of each rectangle (assumed constant)

    y_height
        height of each rectangle (assumed constant)

    scale
        tuple that delimits the range of the heatmap scale
    '''

    df_heatmap = df.pivot(index=x_col, columns=y_col, values=value_col)
    if ax is not None:
        sns.heatmap(df_heatmap, annot=True, vmin=scale[0], vmax=scale[1], ax=ax)
    else:
        ax = sns.heatmap(df_heatmap, annot=True, vmin=scale[0], vmax=scale[1])

    return ax


def check_palette_len(color_length):
    if color_length > len(PALETTE):
        raise ValueError('Need more colors in palette!')


if __name__ == '__main__':

    from spta.region.spatial import SpatialRegion
    import pandas as pd

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

    # Index = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    # Cols = ['A', 'B', 'C', 'D']
    # df = pd.DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

    d = {'x': np.repeat(range(0, 50, 10), 4), 'y': np.tile(range(0, 40, 10), 5), 'v': np.random.randn(20)}
    df = pd.DataFrame(data=d)
    print(df)

    ax = plot_pandas_heatmap(df, x_col='x', y_col='y', value_col='v', scale=(-2, 2))
    ax.set_title('Hello')
    plt.show()
