'''
Generate variable ClusteringMetadata given an identifier.
'''

from spta.clustering.kmedoids import kmedoids_metadata_generator
from spta.clustering.regular import regular_metadata_generator


def kmedoids_suites():
    '''
    A dictionary of (ID, kmedoids_suite), where kmedoids_suite can generate a list of kmedoids
    metadata given several combinations of k_values and seed_values.

    Multiple kmedoids analysis can be defined for the same region metadata.
    Used by experiments.kmedoids.kmedoids to find mask and medoids

    See spta.clustering.kmedoids.KmedoidsClusteringMetadata for more info.
    '''

    # add new here
    return {
        'quick': kmedoids_metadata_generator(k_values=range(2, 4), seed_values=range(0, 2)),
        'large': kmedoids_metadata_generator(k_values=range(12, 15), seed_values=range(0, 2)),
        'all': kmedoids_metadata_generator(k_values=range(2, 151), seed_values=range(0, 3)),
        'k18': kmedoids_metadata_generator(k_values=range(18, 19), seed_values=range(0, 1)),
    }


def regular_suites():
    '''
    Holds lists of k values for regular partitioning
    '''

    # add new here
    return {
        'quick': regular_metadata_generator(k_values=range(2, 4)),
        'part1': regular_metadata_generator(k_values=range(2, 96)),
	'part2': regular_metadata_generator(k_values=range(96, 151, 2)),
    }


def get_suite(clustering_type, clustering_suite_id):
    '''
    Get one of the clustering suites.
    '''
    if clustering_type == 'kmedoids':
        clustering_suite = kmedoids_suites()[clustering_suite_id]
    elif clustering_type == 'regular':
        clustering_suite = regular_suites()[clustering_suite_id]

    return clustering_suite



def suite_options():
    '''
    Returns a set of the possible options of either kmedoids of suite, for when either one
    is usable and the clustering algorithm has not been identified yet.
    Not perfect of course...
    '''
    kmedoids_options = set(kmedoids_suites().keys())
    regular_options = set(regular_suites().keys())
    return kmedoids_options.union(regular_options)
