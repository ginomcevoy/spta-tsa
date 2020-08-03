import csv
import numpy as np
import os

from spta.region import Point
from spta.region.base import BaseRegion

from spta.util import fs as fs_util
from spta.util import arrays as arrays_util


class PredictionQueryResult(BaseRegion):
    '''
    A region that represents the result of a prediction query over a prediction region.
    For each point in the prediction region, it contains the following:

    - The forecast series: from forecast_region
    - The test series (TODO move)
    - The generalization error: from generalization_errors
    - The cluster index associated to the cluster for which the point is identified (TODO move)
    '''

    def __init__(self, solver_metadata, distance_measure,
                 forecast_len, is_future, forecast_subregion, test_subregion,
                 error_subregion, prediction_region,
                 partition, spt_region, output_prefix):
        '''
        Constructor, uses a subset of the numpy matrix that backs the forecast region, the subset
        is taken using the prediction region coordinates.
        '''
        super(PredictionQueryResult, self).__init__(forecast_subregion.as_numpy)

        # solver metadata
        self.metadata = solver_metadata
        self.region_metadata = solver_metadata.region_metadata
        self.clustering_metadata = solver_metadata.clustering_metadata
        self.model_params = solver_metadata.model_params
        self.error_type = solver_metadata.error_type

        # query-specific
        self.forecast_len = forecast_len
        self.is_future = is_future
        self.forecast_subregion = forecast_subregion
        self.test_subregion = test_subregion
        self.error_subregion = error_subregion

        # the offset caused by changing coordinates to the prediction region
        self.prediction_region = prediction_region
        self.offset_x, self.offset_y = prediction_region.x1, prediction_region.y1

        # valid for the entire domain region
        self.partition = partition
        self.spt_region = spt_region
        self.output_prefix = output_prefix

    def lines_for_point(self, relative_point):
        '''
        A summary of the prediction result for a given point in the prediction region.
        Uses coordinates relative to the prediction region
        '''

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        lines = [
            self.text_intro_for_domain_point(domain_point),
            self.text_forecast_for_relative_point(relative_point)
        ]

        if not self.is_future:
            lines.append(self.text_test_for_relative_point(relative_point)),

        lines.append(self.text_error_for_relative_point(relative_point))
        return lines

    def text_intro_for_domain_point(self, domain_point):
        '''
        Text describing point in domain coordinates
        '''
        return 'Point: {} (cluster {})'.format(self.absolute_coordinates_of(domain_point),
                                               self.cluster_index_of(domain_point))

    def text_forecast_for_relative_point(self, relative_point):
        '''
        Text describing forecast series, uses prediction region coordinates
        '''
        forecast_str = 'Forecast:'
        if self.is_future:
            forecast_str = 'Forecast (future):'

        return '{:<20} {}'.format(forecast_str, self.prediction_forecast_at(relative_point))

    def text_test_for_relative_point(self, relative_point):
        '''
        Text describing test series, uses prediction region coordinates
        '''
        return '{:<20} {}'.format('Test:', self.prediction_test_at(relative_point))

    def text_error_for_relative_point(self, relative_point):
        return 'Error ({}): {:.3f}'.format(self.error_type,
                                           self.prediction_error_at(relative_point))

    def prediction_forecast_at(self, relative_point):
        '''
        Forecast at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.forecast_subregion.series_at(relative_point)

    def prediction_test_at(self, relative_point):
        '''
        Test series at point, where (0, 0) is the corner of the prediction region.
        Should only be used when is_future is False
        '''
        if self.is_future:
            raise ValueError('Cannot extract test data for out-of-sample')
        return self.test_subregion.series_at(relative_point)

    def prediction_error_at(self, relative_point):
        '''
        Generalization error at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.error_subregion.value_at(relative_point)

    def result_header(self):
        '''
        The header for the result, e.g. CSV
        '''
        header_row = ['Point', 'cluster_index', 'error', 'forecast_series']

        # when in-sample, add the test series
        if not self.is_future:
            header_row.append('test_series')

        return header_row

    def result_tuple_for_point(self, relative_point):
        '''
        A tuple for the result, given a point in the prediction region.
        '''

        # undo the prediction offset to get coordinates in the domain region
        domain_point = self.to_domain_coordinates(relative_point)

        # 1.a. the absolute coordinates of the point
        coords = self.absolute_coordinates_of(domain_point)

        # 1.b. the index of the cluster that contains the point
        cluster_index = self.cluster_index_of(domain_point)

        # 1.c. generalization error at the point
        error = self.prediction_error_at(relative_point)
        error_str = '{:.3f}'.format(error)

        # 1.d. the forecasted series at the point
        forecast = self.prediction_forecast_at(relative_point)

        point_tuple = [coords, cluster_index, error_str, forecast]

        # 1.e. (is_future False only) the test series at the point
        if not self.is_future:
            point_tuple.append(self.prediction_test_at(relative_point))

        return point_tuple

    def summary_header(self):
        '''
        The header for the summary, e.g. when exported via CSV
        '''
        # the clustering algorithm may have many columns
        clustering_dict = self.clustering_metadata.as_dict()
        clustering_header = clustering_dict.keys()

        csv_header = list(clustering_header)
        csv_header.extend(['clusters', 'mse'])
        return csv_header

    def summary_tuple(self):
        '''
        A tuple that summarizes this result, e.g. when exported via CSV
        '''
        # 2.a. clustering columns, e.g. k, seed
        clustering_data = self.clustering_metadata.as_dict().values()

        # 2.b. number of clusters that intersect the prediction region
        unique_clusters = self.summary_unique_clusters()
        cluster_count = len(unique_clusters)

        # 2.c. MSE of the generalization errors in the prediction result
        mse_error = self.summary_generalization_mse()
        mse_error_str = '{:.3f}'.format(mse_error)

        # write a single row with this data
        summary_list = list(clustering_data)
        summary_list.extend([str(cluster_count), mse_error_str])

        return tuple(summary_list)

    def summary_unique_clusters(self):
        '''
        Returns a list of the clusters that intersect the prediction region
        '''

        # to find the cluster indices for points in the prediction region,
        # convert points to domain coordinates
        domain_points = [
            self.to_domain_coordinates(relative_point)
            for relative_point
            in self
        ]
        clusters_each_point = [
            self.cluster_index_of(domain_point)
            for domain_point
            in domain_points
        ]

        # find the unique indices
        return list(set(clusters_each_point))

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

    def cluster_index_of(self, domain_point):
        '''
        The index of the cluster that has this point as member
        '''
        # return the value of the mask at this point
        return self.partition.membership_of_points([domain_point])[0]

    def absolute_coordinates_of(self, domain_point):
        '''
        Computes the original coordinates of the raw dataset. This takes into consideration
        that the domain region is a subset of the complete raw dataset! Uses the region metadata.
        '''
        # get the absolute coordinates, because the domain region can be a subset too
        return self.region_metadata.absolute_position_of_point(domain_point)

    def save_as_csv(self):
        '''
        Writes two CSV files:
        1. The prediction results (query-result-[...]), one tuple for each point in the prediction
           query, and with these columns:
            1.a. the absolute coordinates of the point
            1.b. the index of the cluster that contains the point
            1.c. generalization error at the point
            1.d. the forecasted series at the point
            1.e. (is_future False only) the test series at the point

           The name of the CSV indicates the forecast length and is_future flag.

        2. The prediction summary (query-summary-[...]), a single tuple with the following info:
            2.a. clustering columns, e.g. k, seed
            2.b. number of clusters that intersect the prediction region
            2.c. MSE of the generalization errors in the prediction result

        The get_csv_file_path(prefix) method is used to calculate the file paths.
        '''

        # ensure output dir
        csv_dir = self.metadata.csv_dir(self.output_prefix, self.prediction_region)
        fs_util.mkdir(csv_dir)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        self.write_prediction_results(csv_dir)
        self.write_prediction_summary(csv_dir)

    def write_prediction_results(self, csv_dir):

        # take of future flag here... ugly but necessary because summary is independant of flag
        future_str = ''
        if self.is_future:
            future_str = '-future'
        result_prefix = 'query-result{}'.format(future_str)

        # write the prediction results
        result_csv_filename = self.metadata.csv_filename(result_prefix, self.prediction_region,
                                                         self.forecast_len)
        result_csv_file = os.path.join(csv_dir, result_csv_filename)
        with open(result_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # first header, then tuple for each point
            csv_writer.writerow(self.result_header())
            for relative_point in self:
                csv_writer.writerow(self.result_tuple_for_point(relative_point))

        self.logger.info('Wrote CSV of prediction result at: {}'.format(result_csv_file))

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
