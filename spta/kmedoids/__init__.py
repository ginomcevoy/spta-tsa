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
