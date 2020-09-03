'''
TODO doco
'''

import argparse

from experiments.metadata.region import predefined_regions
from experiments.metadata.clustering import get_suite, suite_options

from spta.clustering.min_distance import FindClusterWithMinimumDistance
from spta.distance.dtw import DistanceByDTW

from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Extraction LSTM...'

    usage = '%(prog)s [-h] <region> [kmedoids|regular] <clustering_suite> [--random=1000] ' \
        '[--random-seed=0] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-extraction-lstm', description=desc,
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

    # random parameters
    help_msg = 'Number of random points for extraction (default: %(default)s)'
    parser.add_argument('--random', help=help_msg, default=1000, type=int)

    help_msg = 'Random seed for extraction (default: %(default)s)'
    parser.add_argument('--random-seed', help=help_msg, default=0, type=int)

    # logging
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()

    logger = log_util.setup_log_argparse(args)
    analyze_suite(args, logger)


def analyze_suite(args, logger):

    region_metadata, clustering_suite = metadata_from_args(args)

    # default...
    output_home = 'outputs'

    # TODO assuming DTW
    distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    # TODO this code is broken if we don't use DTW
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    min_distance_finder = FindClusterWithMinimumDistance(region_metadata, distance_measure,
                                                         clustering_suite)

    suite_result = min_distance_finder.retrieve_suite_result_csv(output_home)
    min_distance_finder.evaluate_medoid_distance_of_random_points(count=args.random,
                                                                  random_seed=args.random_seed,
                                                                  suite_result=suite_result,
                                                                  output_home=output_home)


def metadata_from_args(args):
    '''
    Extract experiment metadata from request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata
    # FIXME need to change metadata.clustering so that the identifier is passed to the suite
    # here we pass it manually
    clustering_suite = get_suite(args.clustering_type, args.clustering_suite)
    clustering_suite.identifier = args.clustering_suite

    # # a meaningful description to use in the CSV name
    # suite_desc = '{}-{}'.format(args.clustering_type, args.clustering_suite)

    return region_metadata, clustering_suite  # , suite_desc


if __name__ == '__main__':
    processRequest()
