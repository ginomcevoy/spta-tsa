'''
Execute this program to run K-medoids on a dataset, save medoid info as CSV and create variance
plots.
'''
import argparse
import csv
import numpy as np

from experiments.metadata.region import predefined_regions
from experiments.metadata.kmedoids import kmedoids_suites
# from spta.kmedoids.silhouette import KmedoidsWithSilhouette

from spta.distance.dtw import DistanceByDTW
from spta.distance.variance import DistanceHistogramClusters
from spta.kmedoids import kmedoids, medoids_to_absolute_coordinates
from spta.region.partition import PartitionRegionCrisp

from spta.util import fs as fs_util
from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Run k-medoids on a spatio temporal region with different k/seeds'
    usage = '%(prog)s [-h] <region> <kmedoids_id> [--variance] [--random] [--bins] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-kmedoids', description=desc, usage=usage)

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
    do_kmedoids(args, logger)


def do_kmedoids(args, logger):

    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the region and transform to list of time series
    spt_region = region_metadata.create_instance()
    series_group = spt_region.as_2d

    # prepare the CSV output now (header)
    # name is built based on region and kmedoids IDs
    fs_util.mkdir('csv')
    csv_filename = 'csv/cluster_kmedoids_{}_{}.csv'.format(args.region, args.kmedoids_id)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(['k', 'seed', 'total_cost', 'medoids'])

    # retrieve the kmedoids suite
    kmedoids_suite = kmedoids_suites()[args.kmedoids_id]

    # Assumption: assuming DTW
    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(region_metadata.distances_filename,
                                         region_metadata.region)

    # iterate the suite
    for kmedoids_metadata in kmedoids_suite:

        # assumption: overwrite distance_measure...
        # TODO think of a better approach, probably handle this internally to distance_measure
        kmedoids_metadata.distance_measure.distance_matrix = distance_dtw.distance_matrix

        # run K-medoids, this generates a KmedoidsResult namedtuple
        kmedoids_result = kmedoids.run_kmedoids_from_metadata(series_group, kmedoids_metadata)
        k, random_seed = kmedoids_metadata.k, kmedoids_metadata.random_seed

        total_cost = '{:.3f}'.format(kmedoids_result.total_cost)
        medoid_coords = medoids_to_absolute_coordinates(spt_region, kmedoids_result.medoids)
        partial_result = [k, random_seed, total_cost, medoid_coords]

        # write partial result
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            logger.info('Writing partial result: {}'.format(partial_result))
            csv_writer.writerow(partial_result)

        # do variance analysis and plots?
        if args.variance:

            # random points as integer, None means no random points are added
            random_points = None
            if args.random:
                random_points = int(args.random)

            do_variance_analysis(spt_region, distance_dtw, kmedoids_metadata, kmedoids_result,
                                 random_points, args.bins, logger)

    logger.info('CSV output at: {}'.format(csv_filename))


def do_variance_analysis(spt_region, distance_measure, kmedoids_metadata, kmedoids_result,
                         random_points, bins, logger):

    # create spatio-temporal clusters with the labels obtained by k-medoids
    clusters = []
    k, random_seed = kmedoids_metadata.k, kmedoids_metadata.random_seed
    members, centroids = kmedoids_result.labels, kmedoids_result.medoids
    logger.debug('members: {}'.format(members))
    logger.debug('medoids: {}'.format(centroids))

    # short name for plot
    # be nice and pad with zeros if needed
    padding = int(k / 10) + 1
    cluster_name_str = 'cluster{{:0{}d}}'.format(padding)

    # build the spatio-temporal clusters and pass the medoids as centroids
    _, x_len, y_len = spt_region.shape
    partition = PartitionRegionCrisp.from_membership_array(kmedoids_result.labels, x_len, y_len)
    clusters = partition.create_all_spt_clusters(spt_region,
                                                 centroid_indices=kmedoids_result.medoids)

    # iterate clusters to set names and calculate costs
    total_cost = 0
    for i in range(0, k):
        cluster_i = clusters[i]
        cluster_i.name = cluster_name_str.format(i)

        # verification: compute the intra-cluster cost for each cluster
        # should be the same as output from k-medoids
        distances_i = distance_measure.distances_to_point(cluster_i, cluster_i.get_centroid(),
                                                          cluster_i.all_point_indices)
        intra_cluster_cost_i = np.sum(distances_i)
        total_cost += intra_cluster_cost_i
        logger.debug('Intra-cluster cost cluster_{} = {:.3f}'.format(i, intra_cluster_cost_i))

    logger.debug('Total cost = {:.3f}'.format(total_cost))

    # build a suitable name for the current plot: need info on region, k, seed
    plot_name = 'plots/variance_kmedoids_{}_{}_{}.pdf'.format(spt_region.region_metadata,
                                                              k, random_seed)

    # perform the variance analysis
    DistanceHistogramClusters.cluster_histograms(clusters=clusters,
                                                 distance_measure=distance_measure,
                                                 random_points=random_points,
                                                 bins=bins,
                                                 with_statistics=True,
                                                 plot_name=plot_name,
                                                 alpha=0.5)
    # variance.variance_analysis_clusters(clusters, distance_measure, plot_name=plot_name)


if __name__ == '__main__':
    processRequest()
