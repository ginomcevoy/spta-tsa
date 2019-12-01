import numpy as np
import logging

from tslearn.metrics import dtw

from . import region
from . import util


class DistanceWithDTW:

    def apply(self, first_series, second_series):
        return dtw(first_series, second_series)

    def combine_distances(self, distances_for_point):
        '''
        Given many distances, combine them to provide a single metric for the distance between
        one series and a list of series.
        For DTW, we use the Root Sum Squared.
        '''
        return util.root_sum_squared(distances_for_point)


class CalculateCentroid:

    def __init__(self, distance_class=DistanceWithDTW):
        '''
        Use DistanceWithDTW as means of calculating distances between series, by default.
        '''
        self.distance_obj = distance_class()
        self.log = logging.getLogger()

    def distances_from_point_series(self, spatio_temporal_region, point):
        '''
        Given a spatio_temporal_region and a point, computes the distance between the series of
        that point and all the other series.
        '''
        point_series = spatio_temporal_region.series_at(point)
        spatio_temporal_region_as_list = spatio_temporal_region.as_list
        return self.distances_from_point_series_internal(spatio_temporal_region,
                                                         spatio_temporal_region_as_list,
                                                         point, point_series)

    def distances_from_point_series_internal(self, spatio_temporal_region,
                                             spatio_temporal_region_as_list,
                                             point, point_series):
        '''
        This internal methods is useful to avoid converting the sptr to a list every time we
        iterate over a point.
        '''

        # find the distance between the point series and every other series in the region
        # use the distance function provided
        distances_for_point = [
            self.distance_obj.apply(point_series, other_series)
            for other_series
            in spatio_temporal_region_as_list
        ]

        # don't evaluate the series with itself
        # (x_len, y_len, _) = spatio_temporal_region.shape
        # # point_index = point.x * x_len + point.y
        # point_index = point.y * y_len + point.x
        # distances_for_point[point_index] = np.nan

        # TODO consider building a region here

        # np.nanmin()

        return distances_for_point

    def find_point_with_least_distance(self, spatio_temporal_region):
        '''
        Find the point (and its series) that minimizes the combined distance between its series and
        all the other series in a region. For this, we iterate over each point in the region,
        and calculate the combined distance of each point against the whole region.
        '''
        (x_len, y_len, series_len) = spatio_temporal_region.shape

        # if we use the internal function, we only need to do this once
        spatio_temporal_region_as_list = spatio_temporal_region.as_list

        # here we will store all the distances in the region as 2d array
        combined_distances = np.empty((x_len, y_len))

        self.log.info('Calculating centroid of region: %s', (x_len, y_len, series_len))

        # iterate over all points
        for x in range(0, x_len):
            for y in range(0, y_len):
                point = region.Point(x, y)
                point_series = spatio_temporal_region.series_at(point)
                distances_for_point = self.distances_from_point_series_internal(
                    spatio_temporal_region, spatio_temporal_region_as_list,
                    point, point_series)

                self.log.debug('x %s y %s distances: %s' % (x, y, distances_for_point))
                combined_distances[x, y] = self.distance_obj.combine_distances(distances_for_point)

        # print(combined_distances)

        # just find the smallest value now
        minimum, index_2d = util.minimum_value_and_index(combined_distances)
        point = region.Point(index_2d[0], index_2d[1])
        print('%s %s' % (minimum, point))
        return point


if __name__ == '__main__':
    sptr = region.SpatioTemporalRegion.load_4years()
    small_region = sptr.get_small()

    centroid_calc = CalculateCentroid()
    point = centroid_calc.find_point_with_least_distance(small_region)
    print('centroid %s' % str(point))
