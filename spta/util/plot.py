import matplotlib.pyplot as plt


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

    palette = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if len(set(colors)) > len(palette):
        raise ValueError('Need more colors in palette!')

    plt.figure()

    x = range(0, series_len)
    for (index, series) in enumerate(series_group):
        plt.plot(x, series, c=palette[colors[index]])

    plt.show()
