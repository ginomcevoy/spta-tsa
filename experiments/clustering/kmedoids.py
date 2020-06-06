'''
Execute this program to run K-medoids on a dataset, save the medoids as CSV and save
the clusters as npy objects.
'''
import argparse
import csv

from experiments.metadata.region import predefined_regions
from experiments.metadata.kmedoids import kmedoids_suites
# from spta.kmedoids.silhouette import KmedoidsWithSilhouette

from spta.distance.dtw import DistanceByDTW
from spta.kmedoids import kmedoids, get_medoid_indices
from spta.region.temporal import SpatioTemporalRegion

from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Run k-medoids on a spatio temporal region with different k/seeds'
    usage = '%(prog)s [-h] <region> <kmedoids_id>'
    parser = argparse.ArgumentParser(prog='kmedoids', description=desc, usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    kmedoids_options = kmedoids_suites().keys()
    parser.add_argument('kmedoids_id', help='ID of the kmedoids analysis',
                        choices=kmedoids_options)
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    do_kmedoids(args, logger)


def do_kmedoids(args, logger):

    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the region and transform to list of time series
    spt_region = SpatioTemporalRegion.from_metadata(region_metadata)
    series_group = spt_region.as_2d

    # prepare the CSV output now (header)
    # name is built based on region and kmedoids IDs
    csv_filename = '{}_{}.csv'.format(args.region, args.kmedoids_id)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        csv_writer.writerow(['k', 'seed', 'medoids', 'total_cost'])

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
        result = kmedoids.run_kmedoids_from_metadata(series_group, kmedoids_metadata)

        total_cost = '{:.3f}'.format(result.total_cost)
        medoid_coords = medoids_to_absolute_coordinates(spt_region, result.medoids)
        partial_result = [result.k, result.random_seed, medoid_coords, total_cost]

        # write partial result
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            logger.info('Writing partial result: {}'.format(partial_result))
            csv_writer.writerow(partial_result)


def medoids_to_absolute_coordinates(spt_region, medoids):
    '''
    Given Medoid instances, return the absolute coordinates from the original dataset.
    Uses the metadata to recover the subregion that was used to slice the original dataset.
    Returns a list of Point instances.
    '''
    # the indices, relative to the subregion
    medoid_indices = get_medoid_indices(medoids)

    # metadata is used to recover original coordinates
    spt_metadata = spt_region.region_metadata

    # iterate to get coordinates
    coordinates = []
    for medoid_index in medoid_indices:
        medoid_point = spt_metadata.index_to_absolute_points(medoid_index)
        coordinates.append('({}, {})'.format(medoid_point.x, medoid_point.y))

    return coordinates


if __name__ == '__main__':
    processRequest()
