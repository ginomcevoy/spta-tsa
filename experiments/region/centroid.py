'''
Calculate the centroid of a spatio-temporal region. Requires a distance matrix.
To save the centroid, add it manually to experiments.metadata.centroid
'''

import argparse

from experiments.metadata.region import predefined_regions

from spta.region.centroid import CalculateCentroid
from spta.distance.dtw import DistanceByDTW

from spta.util import log as log_util


def processRequest():

    # parses the arguments
    desc = 'Calculate the centroid of a spatio-temporal region.'
    usage = '%(prog)s [-h] <region> [--log=<log_level>]'
    parser = argparse.ArgumentParser(prog='centroid', description=desc, usage=usage)

    # need name of region metadata
    # distance is optional, defaults to DistanceByDTW
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')
    # TODO: optional distance

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    calculate_centroid_with_args(args)


def calculate_centroid_with_args(args):

    # get the region from predefined metadata
    spt_region_metadata = predefined_regions()[args.region]

    # TODO allow other distance measures
    distance_measure = DistanceByDTW()

    # calculate centroid
    CalculateCentroid.for_sptr_metadata(spt_region_metadata, distance_measure)


if __name__ == '__main__':
    processRequest()
