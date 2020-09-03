'''
Execute this program to run K-medoids on a dataset, save medoid info as CSV and create variance
plots.
'''
import argparse
import csv
import numpy as np
import os

from experiments.metadata.region import predefined_regions
from experiments.metadata import clustering
from experiments.metadata.clustering import get_suite

from spta.clustering.factory import ClusteringFactory
from spta.distance.dtw import DistanceByDTW

from spta.util import log as log_util
from spta.util import fs as fs_util


def processRequest():

    # parses the arguments
    desc = 'Perform k-medoids or regular clustering, then save distances between medoids'
    usage = '%(prog)s [-h] <region> [kmedoids|regular] <clustering_suite> [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-medoid-distances', description=desc,
                                     usage=usage)

    # required argument: region ID
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # required argument: clustering algorithm
    clustering_options = ('kmedoids', 'regular')
    parser.add_argument('clustering', help='Name of clustering algorithm',
                        choices=clustering_options)

    # required argument: clustering ID
    parser.add_argument('clustering_suite', help='ID of the clustering suite',
                        choices=clustering.suite_options())

    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_medoid_distances(args, logger)


def do_medoid_distances(args, logger):

    region_metadata, clustering_suite = metadata_from_args(args)
    spt_region = region_metadata.create_instance()
    _, x_len, y_len = spt_region.shape

    # TODO assuming DTW
    distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    # TODO this code is broken if we don't use DTW
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    output_home = 'outputs'

    # iterate possible clusterings
    for clustering_metadata in clustering_suite:
        logger.info('Clustering algorithm: {}'.format(clustering_metadata))

        # clustering algorithm to use
        clustering_factory = ClusteringFactory(distance_measure)
        clustering_algorithm = clustering_factory.instance(clustering_metadata)

        # use the clustering algorithm to get the medoids
        partition = clustering_algorithm.partition(spt_region, with_medoids=True)
        medoid_points = partition.medoids

        # the distance matrix uses the point index, calculate point indices of medoids
        medoid_indices = [
            point.x * y_len + point.y
            for point
            in medoid_points
        ]

        # prepare the CSV output
        # CSV regular: outputs/<region>/regular_k<k>/dtw/medoid_distances.csv
        # CSV k-medoids:  outputs/<region>/kmedoids_k<k>_seed<seed>_lite/dtw/medoid_distances.csv
        output_dir = clustering_metadata.output_dir(output_home, region_metadata, distance_measure)
        fs_util.mkdir(output_dir)
        csv_filename = os.path.join(output_dir, 'medoid_distances.csv')

        compute_and_save_medoid_distances(spt_region, distance_measure, clustering_algorithm,
                                          medoid_points, medoid_indices, csv_filename, logger)


def compute_and_save_medoid_distances(spt_region, distance_measure, clustering_algorithm,
                                      medoid_points, medoid_indices, csv_filename, logger):
    '''
    Given a set of medoids, calculate the distances between them using the distance matrix, and
    save to the given CVS. Works for both k-medoids and regular partitioning.
    '''
    _, _, y_len = spt_region.shape
    k = clustering_algorithm.k

    # use the distance matrix to get the distances between the medoids
    medoid_distances = np.empty((k, k))
    for i in range(0, k):

        medoid = medoid_points[i]

        log_msg = 'Finding distances between medoid: {} and the others: {}'
        logger.debug(log_msg.format(medoid, medoid_indices))

        medoid_distances[i, :] = distance_measure.distances_to_point_with_matrix(spt_region,
                                                                                 medoid,
                                                                                 medoid_indices)

    # prepare the header: 'k' '0' '1' ... 'k-1'
    indices = range(0, k)
    indices_str = [str(i) for i in indices]
    header = ['k'] + indices_str

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(header)

        # write the distance matrix to CSV, the diagonal should be 0
        for i in range(0, k):
            distances_str = [
                '{:.3f}'.format(medoid_distance)
                for medoid_distance
                in medoid_distances[i, :]
            ]

            # also add the current cluster
            row = [str(i)] + distances_str
            csv_writer.writerow(row)

    logger.info('Medoid distances for {} -> {}'.format(clustering_algorithm, csv_filename))


def metadata_from_args(args):
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata
    clustering_suite = get_suite(args.clustering, args.clustering_suite)
    return region_metadata, clustering_suite


if __name__ == '__main__':
    processRequest()
