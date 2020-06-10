import numpy as np

from spta.distance.dtw import DistanceByDTW
from spta.util import log as log_util

from .temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata
from . import Region


class CalculateCentroid(log_util.LoggerMixin):

    def __init__(self, distance_measure, use_distance_matrix=True):
        self.distance_measure = distance_measure
        self.use_distance_matrix = use_distance_matrix

    def find_centroid_and_distances(self, spt_region):
        '''
        Given a spatio-temporal region, find the point (and its series) that minimizes the
        sum of distances between its series and all the other series in a region.
        For this, we use the distance matrix, and find the point for which the sum of distances
        is minimized.
        '''
        _, x_len, y_len = spt_region.shape

        # to find the centroid, iterate over all points in the region
        # it is possible to use np.apply_along_axis to find all the distances, but this does
        # cannot work for clusters.
        # combined_distances = np.apply_along_axis(np.sum, axis=1, arr=distance_matrix)

        # see DistanceBetweenSeries.distances_to_point for explanation of why this is needed
        all_point_indices = spt_region.all_point_indices

        # keep track of local minima until global minimum is found
        with_min_distance = None
        min_sum_distances = np.Inf
        min_distances_to_point = np.Inf

        # iterate points in the region, find sum of distances to each point, keep minimum
        for (point, _) in spt_region:

            # the distance measure will find the distances of each point to this specific point
            # this will also work for clusters
            distances_to_point = self.distance_measure.distances_to_point(spt_region, point,
                                                                          all_point_indices,
                                                                          self.use_distance_matrix)
            sum_distances_to_point = np.sum(distances_to_point)

            if sum_distances_to_point < min_sum_distances:
                # found local minimum
                with_min_distance = point
                min_sum_distances = sum_distances_to_point
                min_distances_to_point = distances_to_point

        # found global minimum
        centroid = with_min_distance
        distances_to_centroid = min_distances_to_point

        log_msg = 'Centroid found at {} with minimum sum of distances {:.3f}'
        self.logger.info(log_msg.format(centroid, np.sum(distances_to_centroid)))

        # return the centroid and the distances of all points in region to it
        return centroid, distances_to_centroid

    @classmethod
    def for_sptr_metadata(cls, spt_region_metadata, distance_measure=DistanceByDTW()):
        '''
        Calculate the centroid of the spatio temporal region using provided distance measure.
        If distance matrix is not available, it will be calculated first.

        Uses DTW by default.
        '''
        spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

        # calculate centroid, should reuse a saved distance matrix if it is available
        centroid_calc = CalculateCentroid(distance_measure)
        centroid, _ = centroid_calc.find_centroid_and_distances(spt_region)
        return centroid


if __name__ == '__main__':
    logger = log_util.setup_log('DEBUG')

    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)

    CalculateCentroid.for_sptr_metadata(nordeste_small_md)  # (4, 6) 58.958 for nordeste_small norm
