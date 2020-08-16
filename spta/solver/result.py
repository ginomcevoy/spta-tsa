import csv
import numpy as np
import os

from spta.region import Point
from spta.region.base import BaseRegion
from spta.region.train import SplitTrainingAndTestLast

from spta.util import fs as fs_util
from spta.util import arrays as arrays_util


class PredictionQueryResult(BaseRegion):
    '''
    A region that represents the result of a prediction query over a prediction region.
    For each point in the prediction region, it contains the following:

    - The forecast series: from forecast_subregion
    - The generalization error: from generalization_errors

    This a base class, subclasses depend on whether out-of-sample or in-sample results are
    requested. For in-sample, the test series is added.

    When using clustering, this information is added for each point:
    - The cluster index associated to the cluster for which the point is identified

    '''
    def __init__(self, solver_metadata, forecast_len, forecast_subregion, error_subregion,
                 prediction_region, spt_region, output_home):
        '''
        Constructor, uses a subset of the numpy matrix that backs the forecast region, the subset
        is taken using the prediction region coordinates.
        '''
        super(PredictionQueryResult, self).__init__(forecast_subregion.as_numpy)

        # solver metadata
        self.metadata = solver_metadata
        self.region_metadata = solver_metadata.region_metadata
        self.model_params = solver_metadata.model_params
        self.test_len = solver_metadata.test_len
        self.error_type = solver_metadata.error_type

        # query-specific
        self.forecast_len = forecast_len
        self.forecast_subregion = forecast_subregion
        self.error_subregion = error_subregion

        # the offset caused by changing coordinates to the prediction region
        self.prediction_region = prediction_region
        self.offset_x, self.offset_y = prediction_region.x1, prediction_region.y1

        # valid for the entire domain region
        self.spt_region = spt_region
        self.output_home = output_home

    def lines_for_point(self, relative_point):
        '''
        A summary of the prediction result for a given point in the prediction region.
        Uses coordinates relative to the prediction region
        '''
        raise NotImplementedError

    def text_intro_for_domain_point(self, domain_point):
        '''
        Text describing point in domain coordinates
        '''
        return 'Point: {}'.format(self.absolute_coordinates_of(domain_point))

    def text_error_for_relative_point(self, relative_point):
        return 'Error ({}): {:.3f}'.format(self.error_type,
                                           self.prediction_error_at(relative_point))

    def prediction_forecast_at(self, relative_point):
        '''
        Forecast at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.forecast_subregion.series_at(relative_point)

    def prediction_error_at(self, relative_point):
        '''
        Generalization error at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.error_subregion.value_at(relative_point)

    def result_header(self):
        '''
        The header for the result, e.g. CSV
        '''
        header_row = ['Point', 'error', 'forecast_series']
        return header_row

    def result_tuple_for_point(self, relative_point):
        '''
        A tuple for the result, given a point in the prediction region.
        '''

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        # 1.a. the absolute coordinates of the point
        coords = self.absolute_coordinates_of(domain_point)

        # 1.c. generalization error at the point
        error = self.prediction_error_at(relative_point)
        error_str = '{:.3f}'.format(error)

        # 1.d. the forecasted series at the point
        forecast = self.prediction_forecast_at(relative_point)

        point_tuple = [coords, error_str, forecast]
        return point_tuple

    def summary_header(self):
        '''
        The header for the summary, e.g. when exported via CSV
        Without clustering, only the MSE appears in the summary.
        '''
        return ('mse',)

    def summary_tuple(self):
        '''
        A tuple that summarizes this result, e.g. when exported via CSV.
        Without clustering, only the MSE appears in the summary.
        '''
        # 2.d. MSE of the generalization errors in the prediction result
        mse_error = self.summary_generalization_mse()
        mse_error_str = '{:.3f}'.format(mse_error)

        return (mse_error_str,)

    def summary_generalization_mse(self):
        '''
        The MSE of the forecasts of all points in the prediction region.
        Calculated by finding each error and applying mean squared.
        '''
        generalization_errors = [
            self.prediction_error_at(relative_point)
            for relative_point
            in self
        ]
        return arrays_util.mean_squared(generalization_errors)

    def to_domain_coordinates(self, relative_point):
        return Point(relative_point.x + self.offset_x, relative_point.y + self.offset_y)

    def absolute_coordinates_of(self, domain_point):
        '''
        Computes the original coordinates of the raw dataset. This takes into consideration
        that the domain region is a subset of the complete raw dataset! Uses the region metadata.
        '''
        # get the absolute coordinates, because the domain region can be a subset too
        return self.region_metadata.absolute_position_of_point(domain_point)

    def save_as_csv(self):
        '''
        Writes two CSV files. These descriptions correspond to the case that clustering is used:

        1. The prediction results (query-result-[...]), one tuple for each point in the prediction
           query, and with these columns:
            1.a. the absolute coordinates of the point
            1.b. the index of the cluster that contains the point
            1.c. generalization error at the point
            1.d. the forecasted series at the point
            1.e. (in-sample forecast only) the test series at the point

           The name of the CSV indicates the forecast length and is_out_of_sample flag.

        2. The prediction summary (query-summary-[...]), a single tuple with the following info:
            2.a. clustering columns, e.g. k, seed
            2.b. number of clusters that intersect the prediction region
            2.c. dtw_r_m: a measure of the distances between the *test* series of the relevant
                 medoids and the test series of the prediction region
            2.d. MSE of the generalization errors in the prediction result

        The get_csv_file_path(prefix) method is used to calculate the file paths.
        '''

        # ensure output dir
        csv_dir = self.metadata.csv_dir(self.output_home, self.prediction_region)
        fs_util.mkdir(csv_dir)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        self.write_prediction_results(csv_dir)
        self.write_prediction_summary(csv_dir)

    def write_prediction_results(self, csv_dir):

        # write the prediction results
        result_csv_filename = self.metadata.csv_filename(name_prefix=self.result_prefix(),
                                                         prediction_region=self.prediction_region,
                                                         forecast_len=self.forecast_len)
        result_csv_file = os.path.join(csv_dir, result_csv_filename)
        with open(result_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # first header, then tuple for each point
            csv_writer.writerow(self.result_header())
            for relative_point in self:
                csv_writer.writerow(self.result_tuple_for_point(relative_point))

        self.logger.info('Wrote CSV of prediction result at: {}'.format(result_csv_file))

    def result_prefix(self):
        '''
        Default prefix for the filename of the results CSV
        '''
        return 'query-result'

    def write_prediction_summary(self, csv_dir):

        # write the prediction summary
        summary_csv_filename = self.metadata.csv_filename('query-summary', self.prediction_region,
                                                          self.forecast_len)
        summary_csv_file = os.path.join(csv_dir, summary_csv_filename)
        with open(summary_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            csv_header = self.summary_header()
            csv_writer.writerow(csv_header)

            csv_tuple = self.summary_tuple()
            csv_writer.writerow(csv_tuple)

        self.logger.info('Wrote CSV of prediction summary at: {}'.format(summary_csv_file))


class InSampleResult(PredictionQueryResult):
    '''
    A region that represents the result of a prediction query over a prediction region.
    Creates in-sample forecasts (to be compared with the test series for each point).
    '''

    def __init__(self, solver_metadata, forecast_len, forecast_subregion, test_subregion,
                 error_subregion, prediction_region, spt_region, output_home):

        super(InSampleResult, self).__init__(solver_metadata=solver_metadata,
                                             forecast_len=forecast_len,
                                             forecast_subregion=forecast_subregion,
                                             error_subregion=error_subregion,
                                             prediction_region=prediction_region,
                                             spt_region=spt_region,
                                             output_home=output_home)

        # only in-sample queries show the test data in the result
        self.test_subregion = test_subregion

    def lines_for_point(self, relative_point):
        '''
        A summary of the prediction result for a given point in the prediction region.
        Uses coordinates relative to the prediction region
        '''

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        lines = [
            self.text_intro_for_domain_point(domain_point),
            self.text_forecast_for_relative_point(relative_point),
            self.text_test_for_relative_point(relative_point),
            self.text_error_for_relative_point(relative_point)
        ]

        return lines

    def text_forecast_for_relative_point(self, relative_point):
        '''
        Text describing forecast series, uses prediction region coordinates
        '''
        return '{:<20} {}'.format('Forecast:', self.prediction_forecast_at(relative_point))

    def text_test_for_relative_point(self, relative_point):
        '''
        Text describing test series, uses prediction region coordinates
        '''
        return '{:<20} {}'.format('Test:', self.prediction_test_at(relative_point))

    def prediction_test_at(self, relative_point):
        '''
        Test series at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.test_subregion.series_at(relative_point)

    def result_header(self):
        '''
        The header for the result, e.g. CSV
        Here we add the column for the test series.
        '''
        header_row = super(InSampleResult, self).result_header()
        header_row.append('test_series')

        return header_row

    def result_tuple_for_point(self, relative_point):
        '''
        A tuple for the result, given a point in the prediction region.
        Here we add the test series.
        '''

        point_tuple = super(InSampleResult, self).result_tuple_for_point(relative_point)
        point_tuple.append(self.prediction_test_at(relative_point))

        return point_tuple


class OutOfSampleResult(PredictionQueryResult):
    '''
    A region that represents the result of a prediction query over a prediction region,
    creates out-of-sample forecasts.
    For each point in the prediction region, it contains the following:

    - The forecast series: from forecast_subregion
    - The generalization error: from generalization_errors
    - The cluster index associated to the cluster for which the point is identified (TODO move)
    '''

    def lines_for_point(self, relative_point):
        '''
        A summary of the prediction result for a given point in the prediction region.
        Uses coordinates relative to the prediction region
        '''

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        lines = [
            self.text_intro_for_domain_point(domain_point),
            self.text_forecast_for_relative_point(relative_point),
            self.text_error_for_relative_point(relative_point)
        ]
        return lines

    def text_forecast_for_relative_point(self, relative_point):
        '''
        Text describing forecast series, uses prediction region coordinates.
        Here we add 'future' to denote out-of-sample forecasting.
        '''
        forecast_str = 'Forecast (future):'
        return '{:<20} {}'.format(forecast_str, self.prediction_forecast_at(relative_point))

    def result_prefix(self):
        '''
        Override prefix for the filename of the results CSV, to indicate that the results
        are out-of-sample.
        '''
        return 'query-result-future'


class ResultWithPartition(PredictionQueryResult):
    '''
    Reifies the decorator pattern to add support for clustering to the results.
    '''

    def __init__(self, decorated, partition):

        # inherit these properties from decorated results
        super(ResultWithPartition, self).__init__(solver_metadata=decorated.metadata,
                                                  forecast_len=decorated.forecast_len,
                                                  forecast_subregion=decorated.forecast_subregion,
                                                  error_subregion=decorated.error_subregion,
                                                  prediction_region=decorated.prediction_region,
                                                  spt_region=decorated.spt_region,
                                                  output_home=decorated.output_home)
        self.decorated = decorated

        # clustering-specific
        self.partition = partition
        self.clustering_metadata = self.metadata.clustering_metadata

    def lines_for_point(self, relative_point):
        '''
        A summary of the prediction result for a given point in the prediction region.
        Here, the decorated list of lines is kept, only the first line is changed to add
        clustering information.
        '''
        lines = self.decorated.lines_for_point(relative_point)

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        # switch the first line with this line
        lines[0] = self.text_intro_for_domain_point(domain_point)
        return lines

    def text_intro_for_domain_point(self, domain_point):
        '''
        Description of point, adds cluster information.
        '''
        msg = 'Point: {} (cluster {})'
        return msg.format(self.decorated.absolute_coordinates_of(domain_point),
                          self.cluster_index_of(domain_point))

    def cluster_index_of(self, domain_point):
        '''
        The index of the cluster that has this point as member
        '''
        # return the value of the mask at this point
        return self.partition.membership_of_points([domain_point])[0]

    def result_header(self):
        '''
        The header for the result, e.g. CSV.
        Inserts the cluster_index to the header, at the second position
        '''
        header_row = self.decorated.result_header()
        header_row.insert(1, 'cluster_index')
        return header_row

    def result_tuple_for_point(self, relative_point):
        '''
        A tuple for the result, given a point in the prediction region.
        Inserts the cluster_index of the point to the tuple, at the second position.
        '''
        # the tuple without clustering
        point_tuple = self.decorated.result_tuple_for_point(relative_point)

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        # 1.b. the index of the cluster that contains the point
        cluster_index = self.cluster_index_of(domain_point)
        point_tuple.insert(1, cluster_index)

        return point_tuple

    def summary_header(self):
        '''
        The header for the summary, e.g. when exported via CSV
        Adds columns for clustering metadata.
        '''
        # the clustering algorithm may have many columns
        clustering_dict = self.clustering_metadata.as_dict()
        clustering_header = clustering_dict.keys()
        summary_header_with_clustering = list(clustering_header)

        # add a column to show count of unique clusters
        summary_header_with_clustering.append('clusters')

        # add a column for the combined distances to the medoids
        summary_header_with_clustering.append('dtw_r_m')

        # add the rest of the columns
        summary_header_decorated = self.decorated.summary_header()
        summary_header_with_clustering.extend(summary_header_decorated)
        return tuple(summary_header_with_clustering)

    def summary_tuple(self):
        '''
        A tuple that summarizes this result, e.g. when exported via CSV
        Adds data relevant to clustering
        '''
        # 2.a. clustering columns, e.g. k, seed
        clustering_data = self.clustering_metadata.as_dict().values()

        # 2.b. number of clusters that intersect the prediction region
        unique_clusters = self.summary_unique_clusters()
        cluster_count = len(unique_clusters)

        # 2.c. dtw_r_m: a measure of the distances between the *test* series of the relevant
        #      medoids and the test series of the prediction region
        dtw_r_m = self.summary_distances_between_region_and_medoids()
        dtw_r_m_string = '{:.2f}'.format(dtw_r_m)

        summary_with_clustering = list(clustering_data)
        summary_with_clustering.append(cluster_count)
        summary_with_clustering.append(dtw_r_m_string)

        # add the rest of the summary
        summary_tuple_decorated = self.decorated.summary_tuple()
        summary_with_clustering.extend(summary_tuple_decorated)
        return tuple(summary_with_clustering)

    def summary_unique_clusters(self):
        '''
        Returns a list of the clusters that intersect the prediction region
        '''

        # to find the cluster indices for points in the prediction region,
        # convert points to domain coordinates
        # domain_points = [
        #     self.to_domain_coordinates(relative_point)
        #     for relative_point
        #     in self
        # ]
        # clusters_each_point = [
        #     self.cluster_index_of(domain_point)
        #     for domain_point
        #     in domain_points
        # ]

        # # find the unique indices
        # return list(set(clusters_each_point))

        # delegate to the partition
        return self.partition.find_indices_of_clusters_intersecting_with(self.prediction_region)

    def summary_distances_between_region_and_medoids(self):
        '''
        Returns a measure for the distance (e.g. DTW) between the points in the prediction region
        and the relevant medoids. This is calculated as follows:

        1. Given the prediction region, find the medoids of the clusters that share common points
           (i.e. intersect) with it.
        2. Go to the dataset, and split the points in the prediction region in training/test,
           according to the tp value used in the solver.
        3. For each point in the prediction region, calcualte the distance (e.g. DTW) between the
           its *test* series and the test series of each medoid. This will create a matrix for
           each medoid.
        4. Combine the distances in each matrix (e.g. sum for DTW) to get a value for each medoid.
        5. Use root mean square to obtain a single value that represents the distance between
           the prediction region and the medoids.

        TODO: assuming test_len = forecast_len, time to break this assumption!
        '''

        # For 1.
        # this retrieves the medoids of the relevant clusters, as points
        relevant_medoids = \
            self.partition.find_medoids_of_clusters_intersecting_with(self.prediction_region)
        self.logger.debug('Relevant medoids: {}'.format(relevant_medoids))

        # For 2.
        # it is more convenient to split the entire region into training and test
        # (as opposed to only the prediction region) because we also need to split the series
        # at the medoids

        # TODO: assuming test_len = forecast_len, time to break this assumption!
        splitter = SplitTrainingAndTestLast(self.forecast_len)
        (training_subset, test_subset) = splitter.split(self.spt_region)

        prediction_test_subset = test_subset.region_subset(self.prediction_region)

        # this is a list of the test series in the prediction region
        test_series_list = [
            point_series_tuple[1]
            for point_series_tuple
            in prediction_test_subset
        ]

        # For 3.
        # calculate the distances to each medoid
        distance_measure = self.metadata.distance_measure
        combined_distances_to_medoids = []
        for medoid in relevant_medoids:

            medoid_test_series = test_subset.series_at(medoid)
            self.logger.debug('Test series at medoid {}: {}'.format(medoid, medoid_test_series))

            # this is a list
            ds = distance_measure.compute_distances_to_a_series(medoid_test_series,
                                                                test_series_list)

            ds_str = ['{0:0.2f}'.format(d) for d in ds]
            self.logger.debug('DTWs to medoid {}: {}'.format(medoid, ds_str))

            # For 4.
            # combined all distances to a medoid into a single value
            combined = distance_measure.combine(ds)
            self.logger.debug('Combined DTW to medoid {}: {:.2f}'.format(medoid, combined))

            combined_distances_to_medoids.append(combined)

        # For 5.
        # use RMS
        distances_between_region_and_medoids = \
            arrays_util.root_mean_squared(combined_distances_to_medoids)
        self.logger.debug('dtw_r_m: {:.2f}'.format(distances_between_region_and_medoids))

        return distances_between_region_and_medoids

    def result_prefix(self):
        '''
        Keep the decorated behavior to distinguish the filename
        '''
        return self.decorated.result_prefix()


class PredictionQueryResultBuilder(object):
    '''
    Reifies builder pattern to create instances of PredictionQueryRegion.
    '''

    def __init__(self, solver_metadata, forecast_len, forecast_subregion, test_subregion,
                 error_subregion, prediction_region, spt_region, output_home, is_out_of_sample):

        if is_out_of_sample:
            # out-of-sample
            self.result = OutOfSampleResult(solver_metadata=solver_metadata,
                                            forecast_len=forecast_len,
                                            forecast_subregion=forecast_subregion,
                                            error_subregion=error_subregion,
                                            prediction_region=prediction_region,
                                            spt_region=spt_region,
                                            output_home=output_home)

        else:
            # in-sample
            self.result = InSampleResult(solver_metadata=solver_metadata,
                                         forecast_len=forecast_len,
                                         forecast_subregion=forecast_subregion,
                                         test_subregion=test_subregion,
                                         error_subregion=error_subregion,
                                         prediction_region=prediction_region,
                                         spt_region=spt_region,
                                         output_home=output_home)

    def with_partition(self, partition):
        '''
        Adds support for clustering via the calculated partition.
        '''
        self.result = ResultWithPartition(self.result, partition)
        return self

    def build(self):
        return self.result
