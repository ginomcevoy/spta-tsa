'''
Execute this program to create regular partitions on a dataset, and save the variance plots.
'''
import argparse
import csv

from experiments.metadata.region import predefined_regions
from experiments.metadata.clustering import regular_suites

from spta.clustering.factory import ClusteringFactory

from spta.distance.dtw import DistanceByDTW
from spta.distance.variance import DistanceHistogramClusters

from spta.util import fs as fs_util
from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Create regular partitions on a spatio temporal region with different k values'
    usage = '%(prog)s [-h] <region> <regular_suite_id> [--variance] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-regular', description=desc, usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # required argument: clustering ID
    help_msg = 'ID of the regular clustering suite (see metadata.clustering)'
    parser.add_argument('regular_suite', help=help_msg, choices=regular_suites().keys())

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
    spt_region = region_metadata.create_instance()
    _, x_len, y_len = spt_region.shape

    # retrieve the regular clustering suite
    regular_suite = regular_suites()[args.regular_suite]

    # prepare the CSV output now (header)
    # name is built based on region and kmedoids IDs
    fs_util.mkdir('csv')
    csv_filename = 'csv/cluster_regular_{}_{}.csv'.format(args.region, args.regular_suite)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(['k', 'total_cost', 'medoids'])

    # Assumption: use DTW
    # use pre-computed distance matrix
    distance_measure = DistanceByDTW()
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    # iterate possible regular clusterings
    for clustering_metadata in regular_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # clustering algorithm to use
        clustering_factory = ClusteringFactory(distance_measure)
        clustering_algorithm = clustering_factory.instance(clustering_metadata)

        # work on this regular clustering
        partial_result = do_regular_partition(spt_region, clustering_algorithm, distance_measure,
                                              args)

        # write partial result
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            logger.info('Writing partial result: {}'.format(partial_result))
            csv_writer.writerow(partial_result)

    logger.info('CSV output at: {}'.format(csv_filename))


def do_regular_partition(spt_region, clustering_algorithm, distance_measure, args):
    '''
    Analyze regular partition for a particular value of k.
    '''
    _, x_len, y_len = spt_region.shape
    k = clustering_algorithm.k

    # use the regular algorithm to get a partition and the medoids
    partition, medoid_points = clustering_algorithm.partition(spt_region, with_medoids=True)

    # keep track of total cost of the each cluster by adding the distances to the point
    intra_cluster_costs = []
    clusters = partition.create_all_spt_clusters(spt_region, medoids=medoid_points)

    # format the medoids for CSV output
    medoids_str = ''

    for i in range(0, k):

        cluster_i = clusters[i]

        # recover the original coordinate of the centroid using the region metadata
        absolute_medoid = \
            spt_region.region_metadata.absolute_position_of_point(cluster_i.centroid)
        medoids_str += '({},{}) '.format(absolute_medoid.x, absolute_medoid.y)

        # use the distance matrix to find the intra-cluster cost (sum of all distances of the
        # cluster points to its medoid)
        point_indices_of_cluster_i = cluster_i.all_point_indices
        distances_to_medoid = \
            distance_measure.distances_to_point(spt_region=cluster_i,
                                                point=cluster_i.centroid,
                                                all_point_indices=point_indices_of_cluster_i)

        # add to total cost of this cluster
        intra_cluster_cost = distance_measure.combine(distances_to_medoid)
        intra_cluster_costs.append(intra_cluster_cost)

    # combine intra_cluster costs to obtain total cost
    total_cost = distance_measure.combine(intra_cluster_costs)
    total_cost_str = '{:0.3f}'.format(total_cost)
    partial_result = [k, total_cost_str, medoids_str]

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
