'''
Given a clustering suite, find, for each seed, the a single value of 'k' that represents the 'elbow' of the
intra-cluster distances.

There will be as many elbows as there are seeds in the suite: to find the elbow,
we assume that the cost is a function of k alone, and find the point with the largest
value of the estimated second derivative of the cost.

Currently only k-medoids is supported!
'''

import argparse

from experiments.metadata.region import predefined_regions
from experiments.metadata.clustering import get_suite, kmedoids_suites

from spta.clustering.suite import FindSuiteElbow, OrganizeClusteringSuite
from spta.distance.dtw import DistanceByDTW

from spta.util import log as log_util


def call_parser():

    # parses the arguments
    desc = '''Given a clustering suite, find, for each seed, the a single value of k that represents the elbow of the intra-cluster distances.
Only supports kmedoids!'''

    usage = '%(prog)s [-h] <region> <clustering_suite> <criterion> [--random=1000] ' \
        '[--random-seed=0] [--threshold=0] [--auto-arima=<auto_arima_id>] [--tp=8] [--error=sMAPE] [--log LOG]'
    parser = argparse.ArgumentParser(prog='cluster-extraction-lstm', description=desc,
                                     usage=usage)

    # need name of region metadata and the ID of the kmedoids
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)

    # required argument: clustering ID
    parser.add_argument('clustering_suite', help='ID of the k-medoids clustering suite',
                        choices=kmedoids_suites().keys())

    # logging
    log_options = ('WARN', 'INFO', 'DEBUG')
    log_help_msg = 'log level (default: %(default)s)'
    parser.add_argument('--log', help=log_help_msg, default='INFO', choices=log_options)

    args = parser.parse_args()
    logger = log_util.setup_log_argparse(args)
    analyze_suite(args, logger)


def analyze_suite(args, logger):

    region_metadata, clustering_suite = metadata_from_args(args)
    spt_region = region_metadata.create_instance()

    # default...
    pickle_home = 'pickle'

    # TODO assuming DTW
    distance_measure = DistanceByDTW()

    # use pre-computed distance matrix
    # TODO this code is broken if we don't use DTW
    distance_measure.load_distance_matrix_2d(region_metadata.distances_filename,
                                             region_metadata.region)

    find_elbow = FindSuiteElbow(clustering_suite, spt_region, distance_measure)
    elbows_by_seed = find_elbow.calculate_elbows_kmedoids(pickle_home)

    metadata_organizer = OrganizeClusteringSuite()
    ordered_metadatas_by_seed = metadata_organizer.organize_kmedoids_suite(clustering_suite)

    for seed, elbow in elbows_by_seed.items():
        print('****** Seed {} ******'.format(seed))
        ks = [metadata.k for metadata in ordered_metadatas_by_seed[seed]]
        print('Values of k: {}'.format(ks))
        print('Elbow found for k={}'.format(elbow))


def metadata_from_args(args):
    '''
    Extract experiment metadata from request.
    '''
    # get the region metadata
    region_metadata = predefined_regions()[args.region]

    # get the clustering metadata
    # FIXME need to change metadata.clustering so that the identifier is passed to the suite
    # here we pass it manually
    clustering_suite = get_suite('kmedoids', args.clustering_suite)
    clustering_suite.identifier = args.clustering_suite

    return region_metadata, clustering_suite


if __name__ == '__main__':
    call_parser()
