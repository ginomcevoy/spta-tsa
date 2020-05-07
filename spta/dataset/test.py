import matplotlib.pyplot as plt

from spta.util import plot as plot_util
from spta.distance.dtw import DistanceByDTW, DistanceBySpatialDTW

from spta.region import Region
from spta.region.spatial import SpatialRegion

data = {
    'brian': [
        'raw/distances_brian_1y_1ppd.npy',
        Region(40, 80, 45, 85),
        'plots/distances_0_0_brian.eps',
        'plots/distances_center_brian.eps'
    ],
    'nordeste_small': [
        'raw/distances_nordeste_small_1y_4ppd.npy',
        Region(43, 50, 85, 95),
        'plots/distances_0_0_nordeste_small.eps',
        'plots/distances_center_nordeste_small.eps'
    ],
    'sp_small': [
        'raw/distances_sp_small_1y_4ppd.npy',
        Region(40, 50, 50, 60),
        'plots/distances_0_0_sp_small.eps',
        'plots/distances_center_sp_small.eps'
    ],
    'whole_brazil': [
        'raw/distances_whole_brazil_1y_1ppd.npy',
        Region(20, 100, 15, 95),
        'plots/distances_0_0_whole_brazil.eps',
        'plots/distances_center_whole_brazil.eps'
    ]
}


if __name__ == '__main__':

    import sys
    src = sys.argv[1]

    weight = 0
    if len(sys.argv) > 2:
        weight = int(sys.argv[2])

    if weight:
        print('With weighted DTW, w={}'.format(weight))
        distance_measure = DistanceBySpatialDTW(weight=weight)
    else:
        print('With DTW')
        distance_measure = DistanceByDTW()

    ds = distance_measure.load_distance_matrix_2d(data[src][0], data[src][1])

    x_len, y_len = (data[src][1].x2 - data[src][1].x1, data[src][1].y2 - data[src][1].y1)
    ds_0 = ds[0].reshape((x_len, y_len))
    plot_util.plot_discrete_spatial_region(SpatialRegion(ds_0), 'Distances to point at (0,0)',
                                           labels=False)
    plt.draw
    plt.savefig(data[src][2])
    plt.show()

    center = int(x_len * y_len / 2)
    ds_center = ds[center].reshape((x_len, y_len))
    plot_util.plot_discrete_spatial_region(SpatialRegion(ds_center), 'Distances to center point',
                                           labels=False)
    plt.draw
    plt.savefig(data[src][3])
    plt.show()
