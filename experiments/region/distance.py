'''
Calculate the distance matrix of a spatio-temporal region. This will save the distances
in a file.
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt

from experiments.metadata.region import predefined_regions

from spta.region.spatial import SpatialRegion
from spta.region.temporal import SpatioTemporalRegion
from spta.distance.dtw import DistanceByDTW
from spta.distance.dtw_parallel import DistanceByDTWParallel

from spta.util import log as log_util
from spta.util import plot as plot_util


def processRequest():

    # parses the arguments
    desc = 'Calculate the distance matrix of a spatio-temporal region.'
    usage = '%(prog)s [-h] <region> <distance> [--parallel=<#workers>] [--plots] \
[--log=<log_level>]'
    parser = argparse.ArgumentParser(prog='distances', description=desc, usage=usage)

    # need name of region metadata
    # distance is mandatory here
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    distance_options = ['dtw']
    parser.add_argument('distance', help='Distance measure', choices=distance_options)
    parser.add_argument('--parallel', help='number of parallel workers')
    parser.add_argument('--plots', help='plot distances at P(0, 0) and center point',
                        action='store_true')
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    calculate_distances_with_args(args, logger)


def calculate_distances_with_args(args, logger):

    # get the region from predefined metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

    # only DTW support for now
    if args.distance == 'dtw':
        distance_measure = DistanceByDTW()

    # use parallelization?
    parallel_workers = None
    if args.parallel:
        parallel_workers = int(args.parallel)

    if parallel_workers:
        # use parallelization
        distance_measure_parallel = DistanceByDTWParallel(parallel_workers)
        distance_matrix = distance_measure_parallel.compute_distance_matrix(spt_region)
        logger.debug(str(distance_matrix))

    else:
        # compute distances in one process
        distance_matrix = distance_measure.compute_distance_matrix(spt_region)

    # save to file
    d_filename = spt_region_metadata.distances_filename
    np.save(d_filename, distance_matrix)
    logger.info('Saved distances to {}'.format(d_filename))

    if args.plots:
        show_distances(spt_region, distance_matrix)


def show_distances(spt_region, distance_matrix):
    '''
    Show 2d graphs for the distances to the first and center points.
    Uses the distance matrix to get the distances to point (0, 0) and to the point that
    is at the center of the graph.

    Default filenames for output files:
    plot/distances_0_0_<name>.eps
    plot/distances_center_<name>.eps

    where <name> is the name in the region metadata.
    '''
    (_, x_len, y_len) = spt_region.shape
    metadata = spt_region.region_metadata

    # work on Point at (0, 0)
    distances_0_0_as_region = SpatialRegion(distance_matrix[0].reshape((x_len, y_len)))
    plot_util.plot_discrete_spatial_region(distances_0_0_as_region,
                                           'Distances to point at (0,0)', clusters=False)
    plt.draw()
    distances_0_0_output = 'plots/distances_0_0_{}.eps'.format(metadata.name)
    plt.savefig(distances_0_0_output)
    plt.show()

    # work on Point at center of graph
    center = int(x_len * y_len / 2)
    if x_len % 2 == 0:
        # so far the center variable points to first element of 'center' row, add half row
        center = center + int(y_len / 2)
    distances_center_as_region = SpatialRegion(distance_matrix[center].reshape((x_len, y_len)))
    plot_util.plot_discrete_spatial_region(distances_center_as_region,
                                           'Distances to center point', clusters=False)
    plt.draw()
    distances_center_output = 'plots/distances_center_{}.eps'.format(metadata.name)
    plt.savefig(distances_center_output)
    plt.show()


if __name__ == '__main__':
    processRequest()
