'''
Execute this program to run K-medoids and silhouette analysis on a dataset
'''
import argparse
import logging

from experiments.metadata.region import predefined_regions
from experiments.metadata.silhouette import silhouette_metadata_by_name
from spta.kmedoids.silhouette import KmedoidsWithSilhouette


def processRequest():

    # parses the arguments
    desc = 'Silhouette analysis: run k-medoids on a spatio temporal region'
    usage = '%(prog)s [-h] <region> <command>'
    parser = argparse.ArgumentParser(prog='silhouette', description=desc, usage=usage)

    # for now, need name of region metadata and the command
    # the silhouette analysis will have the same name as the region metadata
    region_options = predefined_regions().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('command', help='One of save|distances|kmedoids|show_distances')

    args = parser.parse_args()
    do_silhouette(args)


def do_silhouette(args):

    # get the region metadata
    region_md = predefined_regions()[args.region]

    # for now reuse the name
    silhouette_md = silhouette_metadata_by_name(region_md.name)

    # do the silhouette analysis
    silhouette_analysis = KmedoidsWithSilhouette(region_md, silhouette_md)
    silhouette_analysis.execute_command(args.command)


if __name__ == '__main__':

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    processRequest()
