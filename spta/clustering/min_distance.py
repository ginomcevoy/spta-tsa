'''
With k-medoids, we find medoids where representative models can be trained (for a solver).
But, a prediction query may only specify the prediction region, --tp and --tf. So the problem
now is to choose the most appropriate clustering given a predictive query.

Here, we explore the space of clustering metadata (e.g. k-medoids) to find the medoids which have
the minimum distance (e.g. DTW) with a given point in the region.

TODO Move some of this logic to a "SuiteResult" class in spta.clustering.suite
TODO Cannot do it now because the logic that creates the CSV is in experiments instead of spta.
'''
import csv
import numpy as np
import os

from spta.region import Point

from spta.util import log as log_util
from spta.util import maths as maths_util

from .factory import ClusteringMetadataFactory


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

        # need the actual region data for this (should have been only its metadata)
        self.spt_region = self.region_metadata.create_instance()

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
                (medoid.x - x_offset) * self.spt_region.y_len + (medoid.y - y_offset)
                for medoid
                in medoids
            ]

            # use the distance matrix
            distances = self.distance_measure.distances_to_point(spt_region=self.spt_region,
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

        msg_str = 'Found global minimum for {}: {} at index [{}] -> {:.2f}'
        msg = msg_str.format(global_min_clustering_repr, global_min_medoid,
                             global_min_cluster_index, global_min_distance)
        self.logger.debug(msg)

        return (global_min_clustering_repr, global_min_cluster_index, global_min_medoid)

    def evaluate_medoid_distance_of_random_points(self, count, random_seed, suite_result,
                                                  output_home):
        '''
        This is part of an attempt at building a LSTM model. The idea is to create a dataset with
        the following tuple structure to train the model:

        <region> <distance> <clustering> <cluster_index> -> <point_series>

        The left part represents a unique medoid M of a cluster, given a clustering algorithm
        calculated under specific conditions of region and distance. The right part is a temporal
        series of a point P the region, where P and M satisfy the relationship of the method
        find_medoid_with_minimum_distance_to_point() above.

        So here we find <count> 'random' points in the region, and for each of these points we
        create the tuple as described. These points are not completely random, we want to avoid
        actual medoids in our search, because the distance will be zero (if P is some M, then
        find_medoid_with_minimum_distance_to_point(P) will return M).

        The workflow is as follows:

        1. Find all medoids indices (extract_all_medoid_indices_from_suite_result)
        2. Get a random sample of all indices in the region, but removing the medoids
        3. For each point P that represents the index, find its medoid_P using
           find_medoid_with_minimum_distance_to_point
        4. Use the clustering_repr to obtain an instance of ClusteringMetadata
        5. Retrieve the instance representation using as_dict
        6. Store the name of the region metadata, the values in as_dict, the cluster index and
           finally the *series* of P, as a row of a CSV.
        '''
        series_len, x_len, y_len = self.spt_region.shape
        factory = ClusteringMetadataFactory()

        # we are saving a series as CSV, here we will use 3 decimal places for each value
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        medoid_indices = extract_all_medoid_indices_from_suite_result(suite_result,
                                                                      self.spt_region)

        # we want some consistency in random process below so we set the seed here
        np.random.seed(seed=random_seed)

        # this creates the random sample as required
        random_indices = maths_util.random_integers_with_blacklist(count, 0, x_len * y_len - 1,
                                                                   blacklist=medoid_indices)
        self.logger.debug('Random indices found with seed {}: {}'.format(random_seed,
                                                                         random_indices))

        # get actual Point instances
        random_points = [
            Point(int(random_index / y_len), random_index % y_len)
            for random_index in random_indices
        ]

        # prepare the output CSV, at the place where the clustering suite stores its CSV
        csv_dir = self.clustering_suite.csv_dir(output_home, self.region_metadata,
                                                self.distance_measure)

        csv_filename_str = 'random_point_dist_medoid__{!r}_count{}_seed{}.csv'
        csv_filename = csv_filename_str.format(self.clustering_suite, count, random_seed)
        csv_filepath = os.path.join(csv_dir, csv_filename)

        # create the CSV
        with open(csv_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # the header depends on clustering type
            header = calculate_csv_header_given_suite_result(suite_result, series_len)
            csv_writer.writerow(header)

            # calculate each tuple
            for random_point in random_points:

                # find the medoid M for the point P
                result = self.find_medoid_with_minimum_distance_to_point(random_point,
                                                                         suite_result,
                                                                         with_matrix=True)
                (global_min_clustering_repr, global_min_cluster_index, global_min_medoid) = result

                # the row elements need to match the header:
                # region_id, <clustering_metadata>, series[0], series[1].... series[x_len]
                region_id = repr(self.region_metadata)
                row = [region_id]

                clustering_metadata = factory.from_repr(global_min_clustering_repr)
                row.extend(list(clustering_metadata.as_dict().values()))
                row.append(global_min_cluster_index)

                random_point_series = self.spt_region.series_at(random_point)
                random_point_series_str = [
                    '{:0.3f}'.format(elem)
                    for elem in random_point_series
                ]
                row.extend(random_point_series_str)

                csv_writer.writerow(row)

        msg = 'Saved evaluate_medoid_distance_of_random_points at {}'
        self.logger.info(msg.format(csv_filepath))


def extract_all_medoid_indices_from_suite_result(suite_result, spt_region):
    '''
    Given the suite_result dictionary built from its CSV, obtain a set of all the medoid indices.
    This is because we want to filter these indices at some point (evaluate_random_points).
    '''
    # get all medoid as points, we don't want repeated so we use a set
    unique_medoid_points = set()
    for clustering_repr, medoids in suite_result.items():
        unique_medoid_points.update(medoids)

    # to get the indices, we need the region
    y_len = spt_region.y_len
    unique_medoid_indices = [
        unique_medoid_point.x * y_len + unique_medoid_point.y
        for unique_medoid_point in unique_medoid_points
    ]

    return unique_medoid_indices


def calculate_csv_header_given_suite_result(suite_result, series_len):
    '''
    We want something like this:

    region_id   type        k   seed    mode    cluster_index s0 s1 ... s(series_len )
    <region>    kmedoids    2   1       lite    1             (..., ..., )

    But for that, we need to know how the clustering metadata looks like. So we grab the
    first element of the suite_result and build (type, k, seed, mode) if the first element
    is kmedoids, or (type, k) for regular.

    This assumes that a suite only has one type!
    '''
    header = ['region_id']

    # here we get the clustering elements for the header
    factory = ClusteringMetadataFactory()
    first_repr = list(suite_result.keys())[0]
    first_clustering_metadata = factory.from_repr(first_repr)
    clustering_header_elems = list(first_clustering_metadata.as_dict().keys())
    header.extend(clustering_header_elems)

    header.append('cluster_index')

    # we want to store each series element in its own column
    series_header_elems = [
        's' + str(i)
        for i in range(0, series_len)
    ]

    header.extend(series_header_elems)
    return header
