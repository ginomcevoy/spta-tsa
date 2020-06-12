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
from spta.region import Point
from spta.region.centroid import CalculateCentroid
from spta.kmedoids import kmedoids
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster
from spta.region.mask import MaskRegionCrisp

from spta.util import log as log_util
from spta.util import fs as fs_util


def processRequest():

    # parses the arguments
    desc = 'Perform k-medoids and regular clusterings, then save distances between medoids'
    usage = '%(prog)s [-h] <region> <kmedoids_id> [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-medoids', description=desc, usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    kmedoids_options = kmedoids_suites().keys()
    parser.add_argument('kmedoids_id', help='ID of the kmedoids analysis',
                        choices=kmedoids_options)
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_medoid_distances(args, logger)


def do_medoid_distances(args, logger):

    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the region and transform to list of time series
    spt_region = SpatioTemporalRegion.from_metadata(region_metadata)

    _, _, y_len = spt_region.shape
    series_group = spt_region.as_2d

    # output CSV kmedoids: csv/<k>/medoid_distances_<region>_kmedoids_<k>_<seed>.csv
    # output CSV regular:  csv/<k>/medoid_distances_<region>_regular_<k>.csv

    # prepare the CSV output now (header)
    # name is built based on region and kmedoids IDs
    # csv_filename = 'cluster_kmedoids_{}_{}.csv'.format(args.region, args.kmedoids_id)
    # with open(csv_filename, 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
    #                             quoting=csv.QUOTE_MINIMAL)
    #     # header
    #     csv_writer.writerow(['k', 'seed', 'total_cost', 'medoids'])

    # retrieve the kmedoids suite
    kmedoids_suite = kmedoids_suites()[args.kmedoids_id]

    # Assumption: assuming DTW
    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(region_metadata.distances_filename,
                                         region_metadata.region)

    last_k = None

    # iterate the suite
    for kmedoids_metadata in kmedoids_suite:

        # new value for k: run regular cluster with this k
        k = kmedoids_metadata.k
        if k != last_k:
            last_k = k
            do_medoid_distances_regular(spt_region, distance_dtw, k, args, logger)

        # assumption: overwrite distance_measure...
        # TODO think of a better approach, probably handle this internally to distance_measure
        kmedoids_metadata.distance_measure.distance_matrix = distance_dtw.distance_matrix

        # run K-medoids, this generates a KmedoidsResult namedtuple
        kmedoids_result = kmedoids.run_kmedoids_from_metadata(series_group, kmedoids_metadata)
        k, random_seed = kmedoids_metadata.k, kmedoids_metadata.random_seed

        # k-medoids works with indices, also convert to points
        medoid_indices = kmedoids_result.medoids
        medoid_points = [
            Point(int(medoid_index / y_len), medoid_index % y_len)
            for medoid_index
            in medoid_indices
        ]

        # prepare the CSV output
        # output CSV regular:  csv/<k>/medoid_distances_<region>_regular_<k>.csv
        output_dir = 'csv/{}'.format(k)
        csv_file_str = '{}/medoid_distances_{}_kmedoids_k{}_seed{}.csv'
        csv_filename = csv_file_str.format(output_dir, args.region, k, random_seed)

        compute_and_save_medoid_distances(spt_region, distance_dtw, medoid_points, medoid_indices,
                                          csv_filename, logger)


def do_medoid_distances_regular(spt_region, distance_measure, k, args, logger):
    '''
    Perform regular partitioning, calculate medoids, calculate distances between medoids, and
    save these distances to a CSV file.
    '''
    _, x_len, y_len = spt_region.shape

    # for regular clustering, the centroid needs to be calculated as medoid
    calculate_centroid = CalculateCentroid(distance_measure)

    # don't store the clusters, only their medoids
    medoids = []

    # regular clustering is achieved by building a regular mask
    for i in range(0, k):
        mask_i = MaskRegionCrisp.with_regular_partition(k, i, x_len, y_len)
        cluster_i = SpatioTemporalCluster(spt_region, mask_i)

        # find the medoid of each cluster
        medoid_i, _ = calculate_centroid.find_centroid_and_distances(cluster_i)
        medoids.append(medoid_i)

    # the distance matrix uses the point index, calculate point indices of medoids
    medoid_indices = [
        point.x * y_len + point.y
        for point
        in medoids
    ]

    # prepare the CSV output
    # output CSV regular:  csv/<k>/medoid_distances_<region>_regular_<k>.csv
    output_dir = 'csv/{}'.format(k)
    fs_util.mkdir(output_dir)
    csv_filename = '{}/medoid_distances_{}_regular_k{}.csv'.format(output_dir, args.region, k)

    compute_and_save_medoid_distances(spt_region, distance_measure, medoids, medoid_indices,
                                      csv_filename, logger)


def compute_and_save_medoid_distances(spt_region, distance_measure, medoid_points,
                                      medoid_indices, csv_filename, logger):
    '''
    Given a set of medoids, calculate the distances between them using the distance matrix, and
    save to the given CVS. Works for both k-medoids and regular partitioning.
    '''
    _, _, y_len = spt_region.shape
    k = len(medoid_indices)

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

    logger.info('Medoid distances for regular partitioning k={} -> {}'.format(k, csv_filename))


if __name__ == '__main__':
    processRequest()
