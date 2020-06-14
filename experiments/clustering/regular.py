'''
Execute this program to create regular partitions on a dataset, and save the variance plots.
'''
import argparse
import csv

from experiments.metadata.region import predefined_regions
from experiments.metadata.kmedoids import kmedoids_suites

from spta.distance.dtw import DistanceByDTW
from spta.distance.variance import DistanceHistogramClusters
from spta.region.centroid import CalculateCentroid
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster
from spta.region.mask import MaskRegionCrisp

from spta.util import fs as fs_util
from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Create regular partitions on a spatio temporal region with different k values'
    usage = '%(prog)s [-h] <region> <kmedoids_id> [--variance] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-regular', description=desc, usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    kmedoids_options = kmedoids_suites().keys()
    parser.add_argument('kmedoids_id', help='ID of the kmedoids analysis',
                        choices=kmedoids_options)

    # variance analysis: --variance to create histograms, --random to add random points
    # bins to specify bins, default='auto'
    parser.add_argument('--variance', help='Perform variance analysis and plot clusters',
                        action='store_true')
    help_msg = 'Add N random points outside each cluster in variance analysis, =0 for auto N'
    parser.add_argument('--random', help=help_msg)
    help_msg = 'bins argument for plt.hist(), (default: %(default)s)'
    parser.add_argument('--bins', help=help_msg, default='auto')

    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()

    # --random requires --variance
    if 'random' in vars(args) and 'variance' not in vars(args):
        parser.error('--random optional argument requires --variance')

    logger = log_util.setup_log_argparse(args)
    analyze_regular_partitions(args, logger)


def analyze_regular_partitions(args, logger):

    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the region and transform to list of time series
    spt_region = SpatioTemporalRegion.from_metadata(region_metadata)
    _, x_len, y_len = spt_region.shape

    # for now, use the kmedoids suite to get k data
    kmedoids_suite = kmedoids_suites()[args.kmedoids_id]

    # prepare the CSV output now (header)
    # name is built based on region and kmedoids IDs
    fs_util.mkdir('csv')
    csv_filename = 'csv/cluster_regular_{}_{}.csv'.format(args.region, args.kmedoids_id)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(['k', 'total_cost', 'medoids'])

    # retrieve the kmedoids suite
    kmedoids_suite = kmedoids_suites()[args.kmedoids_id]

    # Assumption: use DTW
    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(region_metadata.distances_filename,
                                         region_metadata.region)

    # iterate the suite to get the k values
    # the suite also has seeds, no need for them here
    k_set = set()
    for kmedoids_metadata in kmedoids_suite:
        k_set.add(kmedoids_metadata.k)

    # iterate k values
    for k in k_set:

        # work on this k (single partition, no seed)
        partial_result = do_regular_partition(spt_region, distance_dtw, k, args)

        # write partial result
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            logger.info('Writing partial result: {}'.format(partial_result))
            csv_writer.writerow(partial_result)

    logger.info('CSV output at: {}'.format(csv_filename))


def do_regular_partition(spt_region, distance_measure, k, args):
    '''
    Analyze regular partition for a particular value of k.
    '''
    _, x_len, y_len = spt_region.shape

    # for regular clustering, the centroid needs to be calculated
    calculate_centroid = CalculateCentroid(distance_measure)

    # short name for clusters
    # be nice and pad with zeros if needed
    padding = int(k / 10) + 1
    cluster_name_str = 'cluster{{:0{}d}}'.format(padding)

    # keep track of total cost of the each cluster by adding the distances to the point
    intra_cluster_costs = []
    clusters = []

    # format the centroids for CSV output
    centroids_str = ''

    # regular clustering is achieved by building a regular mask
    for i in range(0, k):
        mask_i = MaskRegionCrisp.with_regular_partition(k, i, x_len, y_len)
        cluster_i = SpatioTemporalCluster(spt_region, mask_i)
        cluster_i.name = cluster_name_str.format(i)

        # find the centroid of each cluster
        cluster_i.centroid, cluster_i.distances_to_centroid = \
            calculate_centroid.find_centroid_and_distances(cluster_i)

        # recover the original coordinate of the centroid using the region metadata
        absolute_centroid = \
            spt_region.region_metadata.absolute_position_of_point(cluster_i.centroid)
        centroids_str += '({},{}) '.format(absolute_centroid.x, absolute_centroid.y)

        # add to total cost of this cluster
        intra_cluster_cost = distance_measure.combine(cluster_i.distances_to_centroid)
        intra_cluster_costs.append(intra_cluster_cost)
        clusters.append(cluster_i)

    # combine intra_cluster costs to obtain total cost
    total_cost = distance_measure.combine(intra_cluster_costs)
    total_cost_str = '{:0.3f}'.format(total_cost)
    partial_result = [k, total_cost_str, centroids_str]

    # do variance analysis and plots?
    if args.variance:

        # random points as integer, None means no random points are added
        random_points = None
        if args.random:
            random_points = int(args.random)

        # build a suitable name for the current plot: need info on region, k
        plot_name = 'plots/variance_regular_{}_{}.pdf'.format(spt_region.region_metadata, k)

        # perform the variance analysis
        DistanceHistogramClusters.cluster_histograms(clusters=clusters,
                                                     distance_measure=distance_measure,
                                                     random_points=random_points,
                                                     bins=args.bins,
                                                     with_statistics=True,
                                                     plot_name=plot_name,
                                                     alpha=0.5)

    return partial_result


if __name__ == '__main__':
    processRequest()
