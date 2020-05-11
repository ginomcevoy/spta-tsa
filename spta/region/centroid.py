import numpy as np
import logging

from spta.distance.dtw import DistanceByDTW
from spta.util import log as log_util

from .temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata
from . import Point, Region


class CalculateCentroid(log_util.LoggerMixin):

    def __init__(self, distance_measure):
        self.distance_measure = distance_measure

    def find_point_with_least_distance(self, spt_region):
        '''
        Given a spatio-temporal region, find the point (and its series) that minimizes the
        combined distance between its series and all the other series in a region.
        For this, we use the distance matrix, and find the point for which the combined distance
        is minimized.
        '''
        if self.distance_measure.distance_matrix is None:

            try:
                # can we load a saved distance matrix?
                # this requires the metadata of the region
                self.distance_measure.load_distance_matrix_md(spt_region.metadata)

            except Exception as err:
                self.logger.debug('Could not load saved distance: {}'.format(err))

                # if distance matrix is not available, go ahead and calculate it
                self.distance_measure.compute_distance_matrix(spt_region)

        # now we should have it
        distance_matrix = self.distance_measure.distance_matrix
        assert distance_matrix is not None

        # apply distance_measure.combine on the rows
        combined_distances = np.apply_along_axis(self.distance_measure.combine, axis=1,
                                                 arr=distance_matrix)

        # get the row with the minimum combined distance
        min_row = np.argmin(combined_distances)

        # rebuid the point
        _, _, y_len = spt_region.shape
        with_min_distance = Point(int(min_row / y_len), min_row % y_len)
        min_distance = combined_distances[min_row]

        log_msg = 'Centroid found at {} with minimum distance {:.3f}'
        self.logger.info(log_msg.format(str(with_min_distance), min_distance))

        return with_min_distance

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
        centroid = centroid_calc.find_point_with_least_distance(spt_region)
        return centroid


if __name__ == '__main__':
    log_util.setup_log('DEBUG')

    # region_of_interest = SpatioTemporalRegion.load_4years().get_small()
    # region_of_interest = SpatioTemporalRegion.load_sao_paulo()
    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)

    CalculateCentroid.for_sptr_metadata(nordeste_small_md)
    # Centroid found at Point(x=5, y=4) with minimum distance 8.175
