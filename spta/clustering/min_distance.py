'''
With k-medoids, we find medoids where representative models can be trained (for a solver).
But, a prediction query may only specify the prediction region, --tp and --tf. So the problem
now is to choose the most appropriate clustering given a predictive query.

Here, we explore the space of clustering metadata (e.g. k-medoids) to find the medoids which have
the minimum distance (e.g. DTW) with a given point in the region.
'''
import csv
import numpy as np
import os

from spta.region import Point
from spta.util import log as log_util


class FindClusterWithMinimumDistance(log_util.LoggerMixin):
    '''
    Explore the space of clustering metadata (e.g. k-medoids) to find the medoids which have
    the minimum distance (e.g. DTW) with a given point in the region.

    Requires the output CSV of experiments.clustering.partition_analysis! Need to run this
    experiment to create the file with the specified k-medoids suite.
    '''

    def __init__(self, region_metadata, distance_measure, clustering_suite):
        '''
        Create an instance of this class.

        region_metadata
            instance of spta.region.SpatioTemporalRegionMetadata

        distance_measure
            e.g. DistanceByDTW

        clustering_suite
            instance of spta.clustering.suite.ClusteringSuite
        '''
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.clustering_suite = clustering_suite

    def retrieve_suite_result_csv(self, output_home):
        '''
        This will open the result CSV of analizing a clustering suite, e.g.
        outputs/nordeste_small_2015_2015_1spd/dtw/clustering__kmedoids-quick.csv

        Then it reads, for each tuple, the metadata representation and the list of medoids.
        Returns a dictionary, where each key is a metadata representation and the value is the
        corresponding list of medoids. Given the metadata representation string, the metadata
        instance can be retrieved using ClusteringMetadataFactory.from_repr()
        '''
        result = {}

        # the suite knows where its result should be stored
        csv_filepath = self.clustering_suite.csv_filepath(output_home=output_home,
                                                          region_metadata=self.region_metadata,
                                                          distance_measure=self.distance_measure)

        if not os.path.isfile(csv_filepath):
            raise ValueError('Could not find CSV: {}'.format(csv_filepath))

        # open the CSV, ignore header
        with open(csv_filepath, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            # ignore header
            next(csv_reader)

            for row in csv_reader:
                # the first column is the representation of the clustering, this is used as key
                clustering_repr = row[0]

                # the second column is the total result, don't want this
                # the third column is the list of medoids as string, use this to create the point
                # instances representing the medoids of the clustering
                medoids_str = row[2]

                # string manipulations to retrieve the medoids as Point instances
                medoids_str_elems = medoids_str.split(' ')[:-1]
                medoids_str_coord_pairs = [
                    medoids_str_elem[1:-1].split(',')
                    for medoids_str_elem
                    in medoids_str_elems
                ]
                medoids = [
                    Point(int(medoids_str_coord_pair[0]), int(medoids_str_coord_pair[1]))
                    for medoids_str_coord_pair
                    in medoids_str_coord_pairs
                ]

                # store this tuple
                result[clustering_repr] = medoids

        return result

    def find_medoid_with_minimum_distance_to_point(self, point, suite_result,
                                                   with_matrix=True):
        '''
        Given a point, explore the list of medoids retrieved in retrieve_suite_result_csv()
        to find the medoid with the minimum distance. Uses the distance_measure, and should
        leverage a distance matrix for speed.

        point
            a Point instance, assumed to be located in the region indicated by the region metadata.

        suite_result
            a result from retrieve_suite_result_csv

        with_matrix
            use a pre-calculated distance matrix

        Returns a tuple with the cluster representation, the index, and the actual medoid point:
        (clustering_repr, cluster_index, medoid)

        NOTE: if more than clustering metadata holds the minimum medoid, the first result is kept.
        '''

        # Keep track of each iteration so that we can retrieve this information when the minimum
        # distance is found.
        global_min_clustering_repr = None
        global_min_cluster_index = None
        global_min_medoid = None
        global_min_distance = np.Inf

        # need the actual region data for this (should have been only its metadata)
        spt_region = self.region_metadata.create_instance()

        # PROBLEM: the medoids are in absolute coordinates, but the indices of the
        # distance matrix are in coordinates relative to the region.
        # Use the region metadata to convert the absolute coordinates by removing the offset.
        # TODO improve this someday
        x_offset, y_offset = self.region_metadata.region.x1, self.region_metadata.region.y1
        msg = 'Region offset for {}: ({}, {})'.format(self.region_metadata, x_offset, y_offset)
        self.logger.debug(msg)

        # Iterating only each clustering representation: work with the list of medoids, in order
        # to leverage the DistanceBetweenSeries interface
        for clustering_repr, medoids in suite_result.items():

            # the interface expects a list of indices, we have a list of points
            # here the conversion is done
            # Mind the offset!
            medoid_indices = [
                (medoid.x - x_offset) * spt_region.y_len + (medoid.y - y_offset)
                for medoid
                in medoids
            ]

            # use the distance matrix
            distances = self.distance_measure.distances_to_point(spt_region=spt_region,
                                                                 point=point,
                                                                 all_point_indices=medoid_indices,
                                                                 only_if_precalculated=with_matrix)

            # find the medoid with minimum distance in the current clustering_repr
            current_min_cluster_index = np.argmin(distances)
            current_min_distance = distances[current_min_cluster_index]
            current_min_medoid = medoids[current_min_cluster_index]

            msg_str = 'Found local minimum for {}: {} at index [{}] -> {:.2f}'
            msg = msg_str.format(clustering_repr, current_min_medoid, current_min_cluster_index,
                                 current_min_distance)
            self.logger.debug(msg)

            # update current global minimum?
            if current_min_distance < global_min_distance:
                global_min_clustering_repr = clustering_repr
                global_min_cluster_index = current_min_cluster_index
                global_min_medoid = current_min_medoid
                global_min_distance = current_min_distance

        return (global_min_clustering_repr, global_min_cluster_index, global_min_medoid)
