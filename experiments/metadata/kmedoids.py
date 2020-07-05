'''
Define kmedoids clustering by name, can override metadata
TODO remove after refactoring code that uses this, use .clustering instead

'''
from spta.kmedoids.kmedoids import kmedoids_suite_metadata


def kmedoids_suites():
    '''
    A dictionary of (ID, kmedoids_suite), where kmedoids_suite can generate a list of kmedoids
    metadata given several combinations of k_values and seed_values.

    Multiple kmedoids analysis can be defined for the same region metadata.
    Used by experiments.kmedoids.kmedoids to find mask and medoids

    See spta.kmedoids.kmedoids.KmedoidsMetadata for more info.
    '''

    # add new here
    return {
        'quick': kmedoids_suite_metadata(k_values=range(2, 4), seed_values=range(0, 1)),
        'large': kmedoids_suite_metadata(k_values=range(12, 15), seed_values=range(0, 1)),
        'even': kmedoids_suite_metadata(k_values=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
                                        seed_values=range(0, 3)),
    }
