from collections import namedtuple


Medoid = namedtuple('Medoid', ('index', 'series'))


def get_medoid_indices(medoids):
    return [
        medoid.index
        for medoid
        in medoids
    ]


def get_medoid_series(medoids):
    return [
        medoid.series
        for medoid
        in medoids
    ]


def medoids_to_absolute_coordinates(spt_region, medoid_indices):
    '''
    Given Medoid indices relative to the subregion, return the absolute coordinates from the
    original dataset.
    Uses the metadata to recover the subregion that was used to slice the original dataset.
    Returns a single string
    '''
    # requires metadata
    assert spt_region.region_metadata is not None

    # metadata is used to recover original coordinates
    spt_metadata = spt_region.region_metadata

    # iterate to get coordinates
    coordinates = ''
    for medoid_index in medoid_indices:
        medoid_point = spt_metadata.index_to_absolute_point(medoid_index)
        coordinates += '({},{}) '.format(medoid_point.x, medoid_point.y)

    return coordinates
