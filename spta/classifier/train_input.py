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

from spta.region import Point, Region
from spta.clustering.factory import ClusteringMetadataFactory
from spta.solver.auto_arima import AutoARIMASolverPickler
from spta.solver.metadata import SolverMetadataBuilder

from spta.util import log as log_util
from spta.util import maths as maths_util


CHOICE_CRITERIA = ['min_distance', 'min_error']


class TrainDataWithRandomPoints(log_util.LoggerMixin):
    '''
    This is part of an attempt at building a LSTM model. The idea is to create a dataset with
    the following tuple structure to train the model:

    <region> <distance> <clustering> <cluster_index> -> <point_series>

    The left part represents a unique medoid M of a cluster, given a clustering algorithm
    calculated under specific conditions of region and distance. The right part is a temporal
    series of a point P the region, where M is chosen given P and some choice criterion, using
    a subclass of MedoidChoiceStrategy.

    So here we find <count> 'random' points in the region, and for each of these points we
    create the tuple as described. These points are not completely random, we want to avoid
    actual medoids in our search.

    The workflow is as follows:

    1. Find all medoids indices (extract_all_medoid_indices_from_suite_result)
    2. Get a random sample of all indices in the region, but removing the medoids
    3. For each point P that represents the index, find its medoid_P using
       a subclass of MedoidChoiceStrategy
    4. Use the clustering_repr to obtain an instance of ClusteringMetadata
    5. Retrieve the instance representation using as_dict
    6. Store the name of the region metadata, the values in as_dict, the cluster index and
       finally the *series* of P, as a row of a CSV.
    '''

    def __init__(self, region_metadata, distance_measure):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.spt_region = self.region_metadata.create_instance()

    def get_choice_strategy(self, criterion, output_home, **choice_args):
        choice_strategies = {
            'min_distance': MedoidsChoiceMinDistance(self.region_metadata, self.distance_measure),
            'min_error': MedoidsChoiceMinPredictionError(self.region_metadata, self.distance_measure, output_home, **choice_args)
        }
        return choice_strategies[criterion]

    def evaluate_score_of_random_points(self, clustering_suite, count, random_seed, criterion, output_home,
                                        **choice_args):

        # strategy that will calculate the medoid M given the point P
        choice_strategy = self.get_choice_strategy(criterion, output_home, **choice_args)

        # read suite partitions from previously calculated result
        suite_result = clustering_suite.retrieve_suite_result_csv(output_home=output_home,
                                                                  region_metadata=self.region_metadata,
                                                                  distance_measure=self.distance_measure)

        series_len, x_len, y_len = self.spt_region.shape
        factory = ClusteringMetadataFactory()

        # we are saving a series as CSV, here we will use 3 decimal places for each value
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        medoid_indices = extract_all_medoid_indices_from_suite_result(suite_result,
                                                                      self.spt_region)

        random_points = self.get_random_points_different_to_medoids(count, random_seed, medoid_indices,
                                                                    x_len, y_len)

        # prepare the output CSV for min_distance
        csv_filepath = choice_strategy.csv_filepath(output_home=output_home,
                                                    clustering_suite=clustering_suite,
                                                    count=count,
                                                    random_seed=random_seed)

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
                result = choice_strategy.choose_medoid(suite_result, random_point, **choice_args)
                (global_min_clustering_repr, global_min_cluster_index, global_min_medoid) = result

                # the row elements need to match the header:
                # region_id, <clustering_metadata>, <medoid_data>, series[0], series[1].... series[x_len]
                region_id = repr(self.region_metadata)
                row = [region_id]

                # clustering_metadata
                clustering_metadata = factory.from_repr(global_min_clustering_repr)
                row.extend(list(clustering_metadata.as_dict().values()))

                # medoid data
                row.append(global_min_cluster_index)
                row.append(global_min_medoid.x)
                row.append(global_min_medoid.y)

                random_point_series = self.spt_region.series_at(random_point)
                random_point_series_str = [
                    '{:0.3f}'.format(elem)
                    for elem in random_point_series
                ]
                row.extend(random_point_series_str)

                csv_writer.writerow(row)

        msg = 'Saved evaluate_medoid_distance_of_random_points at {}'
        self.logger.info(msg.format(csv_filepath))

    def get_random_points_different_to_medoids(self, count, random_seed, medoid_indices, x_len, y_len):

        # we want some consistency in random process below so we set the seed here
        np.random.seed(seed=random_seed)

        # this creates the random sample as required
        max_index = x_len * y_len - 1
        random_indices = maths_util.random_integers_with_blacklist(count, 0, max_index,
                                                                   blacklist=medoid_indices)
        self.logger.debug('Random indices found with seed {}: {}'.format(random_seed,
                                                                         random_indices))

        # get actual Point instances
        random_points = [
            Point(int(random_index / y_len), random_index % y_len)
            for random_index in random_indices
        ]

        return random_points


class MedoidsChoiceStrategy(log_util.LoggerMixin):
    '''
    A generic algorithm that chooses a single medoid from suite_result according to a penalization criterion.
    This will choose one of the medoids of one of the clustering partitions as the 'most suitable' for a point P.

    The input must be:
    - the output of clustering_suite.retrieve_suite_result_csv().
    - a point P used as a reference

    Example of suite_result:
        {
            'kmedoids_k2_seed0_lite': [Point(45,86), Point(47,91)],
            'kmedoids_k3_seed0_lite': [Point(45,86), Point(48,89), Point(45,92)],
            ...
        }
    '''

    def get_medoid_penalties(self, clustering_repr, medoids, point):
        '''
        Implementation of the penalization score, the lower the better for a medoid.
        Should return a list of the scores with the same order as the medoids.
        '''
        pass

    def choose_medoid(self, suite_result, point, **choice_args):
        '''
        Given a point, explore the list of medoids from clustering_suite.retrieve_suite_result_csv()
        to find the medoid with the lowest penalization score.

        point
            a Point instance, assumed to be located in the region indicated by the region metadata.

        suite_result
            a result from retrieve_suite_result_csv

        Returns a tuple with the cluster representation, the index, and the actual medoid point:
        (clustering_repr, cluster_index, medoid)

        NOTE: if more than clustering metadata holds the medoid with the minimum score,
        the first result is kept.
        '''

        # Keep track of each iteration so that we can retrieve this information when the
        # medoid with the lowest penalization is found.
        global_min_clustering_repr = None
        global_min_cluster_index = None
        global_min_medoid = None
        global_min_score = np.Inf

        # Iterating only each clustering representation: work with the list of medoids, in order
        # to leverage the DistanceBetweenSeries interface
        for clustering_repr, medoids in suite_result.items():

            # subclasses will implement this
            scores = self.get_medoid_penalties(clustering_repr, medoids, point)

            # find the medoid with minimum score in the current clustering_repr
            current_min_cluster_index = np.argmin(scores)
            current_min_score = scores[current_min_cluster_index]
            current_min_medoid = medoids[current_min_cluster_index]

            msg_str = 'Found local minimum for {}: {} at index [{}] -> {:.2f}'
            msg = msg_str.format(clustering_repr, current_min_medoid, current_min_cluster_index,
                                 current_min_score)
            self.logger.debug(msg)

            # update current global minimum?
            if current_min_score < global_min_score:
                global_min_clustering_repr = clustering_repr
                global_min_cluster_index = current_min_cluster_index
                global_min_medoid = current_min_medoid
                global_min_score = current_min_score

        # done iterating, we should have the minimum overall score
        msg_str = 'Found global minimum for {}: {} at index [{}] -> {:.2f}'
        msg = msg_str.format(global_min_clustering_repr, global_min_medoid,
                             global_min_cluster_index, global_min_score)
        self.logger.debug(msg)

        return (global_min_clustering_repr, global_min_cluster_index, global_min_medoid)

    def csv_filepath(output_home, clustering_suite, count, random_seed):
        '''
        Returns a proper output filename to store the output of applying this choosing strategy.
        '''
        pass


class MedoidsChoiceMinDistance(MedoidsChoiceStrategy):

    '''
    Given a point, explore the list of medoids to find the medoid with the minimum distance.
    Uses the distance_measure, and requires a pre-calculated distance matrix.
    '''

    def __init__(self, region_metadata, distance_measure):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure

        # PROBLEM: the medoids are in absolute coordinates, but the indices of the
        # distance matrix are in coordinates relative to the region.
        # Use the region metadata to convert the absolute coordinates by removing the offset.
        # TODO improve this someday
        self.x_offset, self.y_offset = region_metadata.region.x1, region_metadata.region.y1
        msg = 'Region offset for {}: ({}, {})'.format(self.region_metadata, self.x_offset, self.y_offset)
        self.logger.debug(msg)

        # need the actual region for this
        self.spt_region = self.region_metadata.create_instance()

    def get_medoid_penalties(self, clustering_repr, medoids, point):
        '''
        Given N, the train input consists of finding N random points in the region that are *not* medoids
        found in a clustering suite. For these N points, find the medoid for which the DTW distance
        between the two time series (random point, medoid), is minimized.
        '''

        # the distances_to_point interface expects a list of indices, we have a list of points
        # here the conversion is done
        # Mind the offset!
        medoid_indices = [
            (medoid.x - self.x_offset) * self.spt_region.y_len + (medoid.y - self.y_offset)
            for medoid
            in medoids
        ]

        # use the distance matrix
        distances = self.distance_measure.distances_to_point(spt_region=self.spt_region,
                                                             point=point,
                                                             all_point_indices=medoid_indices,
                                                             only_if_precalculated=True)

        # the score of each medoid is its distance, we are done
        return distances

    def csv_filepath(self, output_home, clustering_suite, count, random_seed):
        '''
        Returns the CSV filename relevant to minimizing the distance.
        '''
        csv_dir = clustering_suite.csv_dir(output_home, self.region_metadata, self.distance_measure)
        filename = 'random_point_dist_medoid__{!r}_count{}_seed{}.csv'.format(clustering_suite, count, random_seed)
        return os.path.join(csv_dir, filename)


class MedoidsChoiceMinPredictionError(MedoidsChoiceStrategy):

    '''
    Given a point, explore the list of medoids to find the medoid with the minimum distance.
    Uses the distance_measure, and requires a pre-calculated distance matrix.
    '''

    def __init__(self, region_metadata, distance_measure, output_home, model_params, test_len, error_type):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure

        # PROBLEM: the medoids are in absolute coordinates, but the indices of the
        # distance matrix are in coordinates relative to the region.
        # Use the region metadata to convert the absolute coordinates by removing the offset.
        # TODO improve this someday
        self.x_offset, self.y_offset = region_metadata.region.x1, region_metadata.region.y1
        msg = 'Region offset for {}: ({}, {})'.format(self.region_metadata, self.x_offset, self.y_offset)
        self.logger.debug(msg)

        # need the actual region for this
        self.spt_region = self.region_metadata.create_instance()

        self.model_params = model_params
        self.test_len = test_len
        self.error_type = error_type
        self.output_home = output_home

        # cache solvers so that we don't reconstruct them every single time
        self.solvers = {}

    def get_medoid_penalties(self, clustering_repr, medoids, point):
        '''
        Given N, the train input consists of finding N random points in the region that are *not* medoids
        found in a clustering suite. For these N points, find the medoid for which the prediction error
        of the model in the medoid is minimized, when predicting the last test_len elements of the series
        in point.
        '''
        # recover the solver, leverage cache
        solver_metadata = self.build_solver_metadata(clustering_repr)
        solver_repr = repr(solver_metadata)
        self.logger.debug('Looking for solver metadata: {}'.format(solver_repr))
        if solver_repr in self.solvers:
            solver = self.solvers[solver_repr]
            self.logger.debug('Using recovered solver: {}'.format(solver))
        else:
            # new solver
            solver_pickler = AutoARIMASolverPickler(solver_metadata)
            solver = solver_pickler.load_solver()
            self.solvers[solver_repr] = solver
            self.logger.debug('Adding new solver: {}'.format(solver))

        # there are TWO main options here:
        #
        # 1. Use the solver to make calculate the predictions and corresponding errors.
        #    To do this, we use in-sample forecasting which forces test_len (TODO FIXME need better interface here!)
        #    There will only be one usable medoid, the others will get np.Inf as the penalty score.
        #    NOTE: The solver will use the medoid for which the point belongs to according to its partition.
        #
        # 2. Use the solver only to get to the models, but not use it for predictions. Instead, calculate the prediction
        #    error of each medoid and use it as the score.
        #
        # We are here using method 1.

        # start from max penalties, only one medoid will get its proper score which is the prediction error.
        penalties = np.repeat(np.Inf, repeats=len(medoids))

        # build a 1x1 region and ask the solver to calculate the prediction
        # we use forecast_len = 0 to signal in-sample forecasting
        prediction_region = Region(point.x, point.x + 1, point.y, point.y + 1)
        prediction_result = solver.predict(prediction_region, forecast_len=0, output_home=self.output_home, plot=False)

        # recover the error using (0, 0) relative to the prediction region
        forecast_error_at_representative = prediction_result.prediction_error_at(Point(0, 0))

        # which medoid was it? to answer this we need the partition
        # the interface expects a list of points, build a tuple and get only element
        cluster_index_of_representative = solver.partition.membership_of_points((point, ))[0]

        # adjust the penalty of the medoid that represents the point
        penalties[cluster_index_of_representative] = forecast_error_at_representative
        return penalties

    def csv_filepath(self, output_home, clustering_suite, count, random_seed):
        '''
        Returns the CSV filename relevant to minimizing the prediction error of the model in the medoid.
        '''
        csv_dir = clustering_suite.csv_dir(output_home, self.region_metadata, self.distance_measure)
        filename_template = 'random_point_medoid_min_pred_error__{!r}__tp{}__{}__{!r}_count{}_seed{}.csv'
        filename = filename_template.format(self.model_params, self.test_len, self.error_type,
                                            clustering_suite, count, random_seed)
        return os.path.join(csv_dir, filename)

    def build_solver_metadata(self, clustering_repr):
        '''
        With the available information, it is possible to reconstruct the solver metadata
        of the solver with the required models. Once the solver is recovered from persistence,
        the models of the medoids can be accessed, and the prediction error calculated.
        '''
        # recover the clustering metadata from the string
        factory = ClusteringMetadataFactory()
        clustering_metadata = factory.from_repr(clustering_repr)

        builder = SolverMetadataBuilder(self.region_metadata, self.model_params, self.test_len, self.error_type)
        builder.with_clustering(clustering_metadata, self.distance_measure)
        return builder.build()


class MedoidSeriesFormatter(log_util.LoggerMixin):
    '''
    Given a clustering suite, generates a CSV with the following information:

    region_id type k seed mode cluster_index medoid_x medoid_y s0 s1 s2...

    Where:
        region_id identifies a spatio-temporal region
        (type, k, seed, mode) is a clustering metadata (kmedoids)
        (cluster_index, medoid_x, medoid_y) identifies a medoid by its index and coordinates
        (s0, s1, ...) is the temporal series of the medoid for the spatio-temporal region
    '''

    def __init__(self, region_metadata, distance_measure, clustering_suite):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.clustering_suite = clustering_suite
        self.spt_region = self.region_metadata.create_instance()

    def retrieve_medoid_data(self, suite_result, spt_region):
        '''
        Retrieves all the data specified, by using the clustering metadata in each suite element
        and extracting the series in the medoid.
        '''
        # the actual output needs the series, so we will build a new output based on suite_result
        # but with a nested dictionary inside the array:
        # {
        #   clustering_metadata: {
        #      (medoid_list)
        #      [
        #         {
        #            cluster_index: 0,
        #            medoid_x: <x_value0>,
        #            medoid_y: <y_value0>,
        #            series: [s0_0, s0_1, s0_2, ...],
        #         },
        #         {
        #            cluster_index: 1,
        #            medoid_x: <x_value1>,
        #            medoid_y: <y_value1>,
        #            series: [s1_0, s1_1, s1_2, ...],
        #         },
        #      ]
        #   }
        # }
        #
        # (first dictionary)
        #    (medoid_list)
        #       (medoid_dictionary)
        #
        suite_medoid_data = {}

        # PROBLEM: the medoids are in absolute coordinates, but the indices of the
        # distance matrix are in coordinates relative to the region.
        # Use the region metadata to convert the absolute coordinates by removing the offset.
        # TODO improve this someday
        x_offset, y_offset = self.region_metadata.region.x1, self.region_metadata.region.y1
        msg = 'Region offset for {}: ({}, {})'.format(self.region_metadata, x_offset, y_offset)
        self.logger.debug(msg)

        # Iterating each clustering representation, then each medoid in it
        for clustering_repr, medoids in suite_result.items():

            # the first dictionary
            suite_medoid_data[clustering_repr] = []

            # build the medoid list by iterating each medoid
            for i, medoid in enumerate(medoids):

                # for each medoid, build the medoid_dictionary here
                medoid_entry = {}
                medoid_entry['cluster_index'] = i

                # here we don't convert coordinates, expose absolute coordinates to the CSV
                medoid_entry['medoid_x'] = medoid.x
                medoid_entry['medoid_y'] = medoid.y

                # for the series we do need the converted coordinates
                medoid_region_point = Point(medoid.x - x_offset, medoid.y - y_offset)
                medoid_entry['series'] = spt_region.series_at(medoid_region_point)

                suite_medoid_data[clustering_repr].append(medoid_entry)

        return suite_medoid_data

    def produce_csv(self, output_home):
        '''
        Gneerates the CSV
        '''
        suite_result = self.clustering_suite.retrieve_suite_result_csv(output_home=output_home,
                                                                       region_metadata=self.region_metadata,
                                                                       distance_measure=self.distance_measure)

        factory = ClusteringMetadataFactory()

        # call helper method
        suite_medoid_data = self.retrieve_medoid_data(suite_result, self.spt_region)

        # prepare the output CSV for min_distance
        csv_filepath = \
            self.clustering_suite.medoid_series_csv_filepath(output_home=output_home,
                                                             region_metadata=self.region_metadata,
                                                             distance_measure=self.distance_measure)

        # create the CSV
        with open(csv_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # the header depends on clustering type
            # it is the same as for TrainDataWithRandomPoints!
            header = calculate_csv_header_given_suite_result(suite_result, self.spt_region.series_len)
            csv_writer.writerow(header)

            # to iterate each tuple, we need to iterate both the clustering metadata and the medoids
            for clustering_repr, medoid_entries in suite_medoid_data.items():

                for medoid_entry in medoid_entries:

                    # the row elements need to match the header:
                    # region_id, <clustering_metadata>, <medoid_data>, series[0], series[1].... series[x_len]
                    region_id = repr(self.region_metadata)
                    row = [region_id]

                    # clustering_metadata
                    clustering_metadata = factory.from_repr(clustering_repr)
                    row.extend(list(clustering_metadata.as_dict().values()))

                    # medoid data
                    row.append(medoid_entry['cluster_index'])
                    row.append(medoid_entry['medoid_x'])
                    row.append(medoid_entry['medoid_y'])

                    medoid_series_str = [
                        '{:0.3f}'.format(elem)
                        for elem in medoid_entry['series']
                    ]
                    row.extend(medoid_series_str)

                    csv_writer.writerow(row)

        msg = 'Saved medoid_data at {}'
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

    # medoid-specific columns
    header.extend(['cluster_index', 'medoid_x', 'medoid_y'])

    # we want to store each series element in its own column
    series_header_elems = [
        's' + str(i)
        for i in range(0, series_len)
    ]

    header.extend(series_header_elems)
    return header
