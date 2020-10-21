from spta.region import Point
from spta.region.centroid import CalculateCentroid
from spta.util import log as log_util

from .region import predefined_regions


def precalculated_centroids():
    '''
    Add centroids here once they have been calculated
    Key is (sptr_name, classname_of_distance_measure)
    '''
    precalculated_centroids = {
        ('nordeste_small_1y_1ppd', 'DistanceByDTW'): Point(4, 6),   # (5, 4) if using rms...
        ('nordeste_small_1y_1ppd_norm', 'DistanceByDTW'): Point(4, 6),
    }
    return precalculated_centroids


def centroid_by_region_and_distance(sptr_name, distance_measure):
    '''
    Returns the centroid of a spatio-temporal region given a distance measure.
    If the centroid has been precalculated and added to the saved centroids, it can be returned
    immediately. Otherwise, it is calculated here.

    sptr_name
        one of the predefined regions in experiments.metadata.region
    distance_measure
        an instance of one of the DistanceBetweenSeries subclasses, e.g. DistanceByDTW()
    '''
    # use a tuple of two strings as the dictionary key
    key = (sptr_name, distance_measure.__class__.__name__)
    centroid = None

    if key in precalculated_centroids():

        centroid = precalculated_centroids()[key]

        logger = log_util.logger_for_me(centroid_by_region_and_distance)
        logger.info('Found precalculated centroid for {}: {}'.format(sptr_name, centroid))

    else:

        # TODO allow other distances
        assert distance_measure.__class__.__name__ == 'DistanceByDTW'

        spt_region_metadata = predefined_regions()[sptr_name]
        centroid = CalculateCentroid.for_sptr_metadata(spt_region_metadata, distance_measure)

    return centroid


if __name__ == '__main__':

    from spta.distance.dtw import DistanceByDTW

    log_util.setup_log('DEBUG')
    centroid = centroid_by_region_and_distance('nordeste_small_1y_1ppd', DistanceByDTW())
    print(centroid)
