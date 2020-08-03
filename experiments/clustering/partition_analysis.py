'''
Execute this program to perform a suite of clustering algorithms (e.g. K-medoids) on a
spatio-temporal region, and analyze the resulting partition:

- sum of intra-cluster costs
- medoid coordinates (original dataset)
- variance analysis (cluster histograms with optional random points outside of the cluster)

The outputs are: a CSV with the costs and medoids for each clustering algorithm in the suite,
and optionally a PDF of the variance histograms.
'''

import argparse
import csv
import os

from experiments.metadata.region import predefined_regions
from experiments.metadata.clustering import get_suite, suite_options

from spta.clustering.factory import ClusteringFactory
from spta.distance.dtw import DistanceByDTW
from spta.distance.variance import DistanceHistogramClusters

from spta.util import fs as fs_util
from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Run a suite of clustering algorithms, calculate intra-cluster costs and optionally ' \
        'create variance histograms'

    usage = '%(prog)s [-h] <region> [kmedoids|regular] <clustering_suite> [--variance] ' \
        '[--random] [--bins] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-partition-analysis', description=desc,
                                     usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # required argument: clustering algorithm
    clustering_options = ('kmedoids', 'regular')
    parser.add_argument('clustering_type', help='Type of clustering algorithm',
                        choices=clustering_options)

    # required argument: clustering ID
    parser.add_argument('clustering_suite', help='ID of the clustering suite',
                        choices=suite_options())

    # variance analysis: --variance to create histograms, --random to add random points
    # bins to specify bins, default='auto'
    parser.add_argument('--variance', help='Perform variance analysis and plot clusters',
                        action='store_true')
    help_msg = 'Add N random points outside each cluster in variance analysis, =0 for auto N'
    parser.add_argument('--random', help=help_msg)
    help_msg = 'bins argument for plt.hist(), (default: %(default)s)'
    parser.add_argument('--bins', help=help_msg, default='auto')

    # logging
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()

    # --random requires --variance
    if 'random' in vars(args) and 'variance' not in vars(args):
        parser.error('--random optional argument requires --variance')

    logger = log_util.setup_log_argparse(args)
    analyze_suite(args, logger)


def analyze_suite(args, logger):

    region_metadata, clustering_suite, suite_desc = metadata_from_args(args)

    # default...
    output_home = 'outputs'

    # TODO assuming DTW
    distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    # TODO this code is broken if we don't use DTW
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    # prepare the CSV output now (header)
    # <output>/<region>/<distance>/clustering__<clustering_type>-<clustering_suite>.csv
    csv_dir = '{}/{!r}/{!r}'.format(output_home, region_metadata, distance_measure)
    fs_util.mkdir(csv_dir)

    csv_filename = 'clustering__{}.csv'.format(suite_desc)
    csv_filepath = os.path.join(csv_dir, csv_filename)

    with open(csv_filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(['clustering', 'total_cost', 'medoids'])

    # iterate possible clusterings
    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # clustering algorithm to use
        clustering_factory = ClusteringFactory(distance_measure)
        clustering_algorithm = clustering_factory.instance(clustering_metadata)

        # work on this clustering
        partial_result = analyze_partition(region_metadata, clustering_algorithm, output_home,
                                           logger, args)

        # write partial result
        with open(csv_filepath, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            logger.info('Writing partial result: {}'.format(partial_result))
            csv_writer.writerow(partial_result)

    logger.info('CSV output at: {}'.format(csv_filepath))


def analyze_partition(region_metadata, clustering_algorithm, output_home, logger, args):

    # recover the regions
    spt_region = region_metadata.create_instance()
    _, x_len, y_len = spt_region.shape

    # use the clustering algorithm to get the partition and medoids
    # will try to leverage pickle and load previous attempts, otherwise calculate and save
    partition = clustering_algorithm.partition(spt_region, with_medoids=True,
                                               save_csv_at=output_home,
                                               pickle_home='pickle')

    distance_measure = clustering_algorithm.distance_measure

    # keep track of total cost of the each cluster by adding the distances to the point
    intra_cluster_costs = []
    clusters = partition.create_all_spt_clusters(spt_region, medoids=partition.medoids)

    # format the medoids for CSV output
    medoids_str = ''

    for i in range(0, clustering_algorithm.k):

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

    # describe the current algorithm, "k" is not enough for k-medoids so go for the repr. string
    clustering_desc = '{!r}'.format(clustering_algorithm)
    partial_result = [clustering_desc, total_cost_str, medoids_str]

    # do variance analysis and plots?
    if args.variance:

        # random points as integer, None means no random points are added
        random_points = None
        if args.random:
            random_points = int(args.random)

        # build a suitable name for the current plot: need info on region, k
        plot_dir = clustering_algorithm.output_dir(output_home, region_metadata)
        fs_util.mkdir(plot_dir)

        # perform the variance analysis
        DistanceHistogramClusters.cluster_histograms(clusters=clusters,
                                                     clustering_algorithm=clustering_algorithm,
                                                     random_points=random_points,
                                                     bins=args.bins,
                                                     with_statistics=True,
                                                     plot_dir=plot_dir,
                                                     alpha=0.5)

    return partial_result


def metadata_from_args(args):
    '''
    Extract experiment metadata from request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata
    clustering_suite = get_suite(args.clustering_type, args.clustering_suite)

    # a meaningful description to use in the CSV name
    suite_desc = '{}-{}'.format(args.clustering_type, args.clustering_suite)

    return region_metadata, clustering_suite, suite_desc


if __name__ == '__main__':
    processRequest()
