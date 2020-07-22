'''
Produces temporal forecasts for points in the region, based on the model trained at medoids.
Uses a clustering algorithm to define clusters and medoids, and uses auto ARIMA for forecasting.
'''
import matplotlib.pyplot as plt

from collections import namedtuple
import csv
import numpy as np
import pickle
import os

from spta.arima.forecast import ArimaModelRegion
from spta.arima import training

from spta.clustering.factory import ClusteringFactory

from spta.region import Point
from spta.region.base import BaseRegion
from spta.region.error import MeasureForecastingError, get_error_func
from spta.region.scaling import SpatioTemporalScaled
from spta.region.spatial import SpatialCluster
from spta.region.train import SplitTrainingAndTestLast

from spta.util import arrays as arrays_util
from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from .metadata import SolverMetadata

# default forecast length
FORECAST_LENGTH = 8


class AutoARIMATrainer(log_util.LoggerMixin):
    '''
    Trains the ARIMA models by partitioning the region and using the ARIMA models at the medoids
    of the resulting clusters
    '''

    def __init__(self, region_metadata, clustering_metadata, distance_measure, auto_arima_params):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.auto_arima_params = auto_arima_params

        # flag
        self.prepared = False

    def prepare_for_training(self):

        # use pre-computed distance matrix
        # TODO this code is broken if we don't use DTW
        self.distance_measure.load_distance_matrix_2d(self.region_metadata.distances_filename,
                                                      self.region_metadata.region)

        # clustering algorithm to use
        clustering_factory = ClusteringFactory(self.distance_measure)
        self.clustering_algorithm = clustering_factory.instance(self.clustering_metadata)

        self.metadata = SolverMetadata(region_metadata=self.region_metadata,
                                       clustering_metadata=self.clustering_metadata,
                                       distance_measure=self.distance_measure,
                                       model_params=self.auto_arima_params)
        self.prepared = True

    def train(self, error_type, training_len=None, test_len=FORECAST_LENGTH,
              output_prefix='outputs'):
        '''
        Partition the region into clusters, then train the ARIMA models on the cluster medoids.

        This code assumes that each temporal series is split into training and test subseries.
        Needs to change when we use a fixed number of points in the past to train the model.
        '''
        if not self.prepared:
            self.prepare_for_training()

        # recover the spatio-temporal region
        spt_region = self.region_metadata.create_instance()

        # create training/test regions
        splitter = SplitTrainingAndTestLast(test_len)
        (training_region, test_region) = splitter.split(spt_region)

        # get the cluster partition and corresponding medoids, save the CSV
        partition, medoids = self.clustering_algorithm.partition(spt_region,
                                                                 with_medoids=True,
                                                                 save_csv_at=output_prefix)

        # partition will be saved for later predictions, go ahead and also save the medoids in it
        partition.medoids = medoids

        self.logger.info('Training solver: {}'.format(self.metadata))

        # use the partition to train ARIMA models at the cluster medoids and use them as
        # representatives for their own clusters. These models will be used to evalute and save
        # the prediction, but will not be the final models.
        # The output is a single spatial region with k different models!
        arima_model_region_for_error = self.train_auto_arima_at_medoids(training_region,
                                                                        partition, medoids)

        # calculate the generalization error for these ARIMA models: it is the prediction error
        # that is calculated using the test dataset.
        # TODO a lot of assumptions here, work them out later
        generalization_errors = self.calculate_errors(arima_model_region_for_error,
                                                      training_region, test_region,
                                                      forecast_len=test_len,
                                                      error_type=error_type)
        self.logger.info('Overall error: {:.4f}'.format(generalization_errors.overall_error))

        # we have the errors, these will be saved
        # now we need to re-train the models at the medoids with the full dataset
        # which will be used for prediction
        arima_model_region_for_predict = self.train_auto_arima_at_medoids(spt_region,
                                                                          partition, medoids)

        # create a solver with the data acquired, this solver can answer queries
        solver = AutoARIMASolver(solver_metadata=self.metadata,
                                 error_type=error_type,
                                 partition=partition,
                                 arima_model_region=arima_model_region_for_predict,
                                 generalization_errors=generalization_errors)
        return solver

    def train_auto_arima_at_medoids(self, training_region, partition, medoids):
        '''
        The ARIMA model region that will be used for forecasting has only k models; the model from
        each medoid will be replicated throughout its cluster.
        This is achieved by creating k clusters, each containing the trained model at its
        medoid, then merging the clusters with the values at the representatives.

        See PartitionRegionCrisp.merge_with_representatives_2d for details.
        '''
        _, x_len, y_len = training_region.shape

        # use partition to create clusters for the training dataset
        # the idea is to use train ARIMA models at the cluster medoids
        training_clusters = partition.create_all_spt_clusters(training_region,
                                                              medoids=medoids)

        # the 'sparse' numpy array that will store the ARIMA models and has the region shape
        # note that all clusters share the same underlying dataset (should be no problem...)
        arima_medoids_numpy = np.empty((x_len, y_len), dtype=object)

        # iterate the training clusters to produce an ARIMA cluster for each one
        # using enumerate to also get the cluster index in [0, k-1]
        arima_clusters = []
        for training_cluster in training_clusters:

            arima_cluster = self.train_auto_arima_at_medoid(partition, training_cluster,
                                                            arima_medoids_numpy)
            arima_clusters.append(arima_cluster)

        # merge the clusters into a single region, this will be backed by a new numpy matrix,
        # that should look exactly like arima_medoids_numpy
        arima_region = partition.merge_with_representatives_2d(arima_clusters, medoids)
        # self.logger.debug('Merged ARIMA region: {}'.format(arima_region.as_numpy))

        # we want an ArimaModelRegion that can be applied to produce forecasts, so we wrap
        # the data with this instance
        return ArimaModelRegion(arima_region.as_numpy)

    def calculate_errors(self, arima_model_region, training_region, test_region, forecast_len,
                         error_type):
        '''
        Calculate the generalization error for these ARIMA models: it is the prediction error
        that is calculated using the test dataset.

        TODO a lot of assumptions here, work them out later
        '''
        # create a forecast: ARIMA requires an empty region (no data is required, only shape)
        # use the test region for convenience
        forecast_region = arima_model_region.apply_to(test_region, forecast_len)

        # calculate the error using a generic error function, requires the error type
        error_func = get_error_func(error_type)
        measure_error = MeasureForecastingError(error_func, test_region, training_region)

        # will calculate the forecast error at each point of the region
        return measure_error.apply_to(forecast_region)

    def train_auto_arima_at_medoid(self, partition, training_cluster, arima_medoids_numpy):
        '''
        Given a spatio-temporal cluster with training data and its medoid, train a single ARIMA
        model at the medoid, and replicate it over the cluster. This effectively turns the medoid
        in the representative point for creating forecasts in the cluster.

        Returns a SpatialCluster that is wrapped around an ArimaModelRegion. This later can be
        used to merge all clusters into a single region built with k representative models.

        The arima_medoids_numpy is passed so that all the clusters are backed by the same
        underlying matrix that holds the k representitave models.
        '''
        # the series to train auto ARIMA at the medoid
        medoid = training_cluster.centroid
        training_series_at_medoid = training_cluster.series_at(medoid)

        # train an ARIMA model and store it at the medoid coordinates
        arima_at_medoid = training.train_auto_arima(self.auto_arima_params,
                                                    training_series_at_medoid)
        arima_medoids_numpy[medoid.x, medoid.y] = arima_at_medoid
        self.logger.debug('Trained ARIMA at medoid {}: {}'.format(medoid, arima_at_medoid))

        # create an ARIMA model region for this cluster
        # but we need a spatial cluster to use merge later! so wrap this region as a cluster
        arima_model_cluster = SpatialCluster(ArimaModelRegion(arima_medoids_numpy), partition,
                                             cluster_index=training_cluster.cluster_index)
        return arima_model_cluster


class AutoARIMASolver(log_util.LoggerMixin):
    '''
    Represents a cluster partition of a given spatio-temporal region, with ARIMA models trained
    at the medoids. It can answer a forecast query of a specified subregion, by using, at each
    point in the subregion, the forecast at the medoid of the cluster for which each point is
    a member.
    '''
    def __init__(self, solver_metadata, error_type, partition, arima_model_region,
                 generalization_errors):

        # user input
        self.metadata = solver_metadata
        self.region_metadata = solver_metadata.region_metadata
        self.clustering_metadata = solver_metadata.clustering_metadata
        self.distance_measure = solver_metadata.distance_measure
        self.auto_arima_params = solver_metadata.model_params

        self.error_type = error_type

        # calculated during training
        self.partition = partition
        self.arima_model_region = arima_model_region
        self.generalization_errors = generalization_errors

        # flag
        self.prepared = False

    def predict(self, prediction_region, forecast_len=FORECAST_LENGTH, output_prefix='outputs',
                plot=True):

        self.logger.debug('Predicting for region: {}'.format(prediction_region))

        if not self.prepared:
            self.prepare_for_predictions()

        # plot the whole region and the prediction region
        self.plot_regions(prediction_region, output_prefix)

        px1, px2, py1, py2 = prediction_region.x1, prediction_region.x2, prediction_region.y1, \
            prediction_region.y2

        # the generalization errors are precalculated, just subset them
        error_subregion = self.generalization_errors.region_subset(prediction_region)

        # compute the prediction: create a forecast over the region
        # to do this, obtain a subset of the model region
        # TODO better support for this...
        arima_models_subset_np = self.arima_model_region.as_numpy[px1:px2, py1:py2]
        arima_model_subset = ArimaModelRegion(arima_models_subset_np)

        # here we just need an empty region with the correct shape
        # and the error_subregion is right there... so use it
        forecast_subregion = arima_model_subset.apply_to(error_subregion, forecast_len)

        # handle descaling here: we want to present descaled data to users
        if self.region_metadata.normalized:

            # The forecast_subregion is not aware of the scaling, because it was not created
            # as such. As a workaround, use the original spatio-temporal region (which HAS the
            # scaling data) and to retrieve appropriate descaling info for this forecast subregion.
            # TODO make this work: forecast_subregion = forecast_subregion.descale()
            self.logger.debug('About to descale manually: {}'.format(forecast_subregion))
            spt_subset_with_scaling_data = self.spt_region.region_subset(prediction_region)
            scaled_forecast_subregion = \
                SpatioTemporalScaled(forecast_subregion,
                                     scale_min=spt_subset_with_scaling_data.scale_min,
                                     scale_max=spt_subset_with_scaling_data.scale_max)
            forecast_subregion = scaled_forecast_subregion.descale()

        # this has all the required information and can be iterated
        return PredictionQueryResult(solver_metadata=self.metadata,
                                     distance_measure=self.distance_measure,
                                     forecast_len=forecast_len,
                                     forecast_subregion=forecast_subregion,
                                     error_subregion=error_subregion,
                                     prediction_region=prediction_region,
                                     partition=self.partition,
                                     spt_region=self.spt_region,
                                     error_type=self.error_type,
                                     output_prefix=output_prefix)

    def prepare_for_predictions(self):
        '''
        Prepare to attend prediction requests. For now, this is called on-demand.
        In a distant future, it can be used to lazily instantiate a solver.
        '''

        # load the region again
        # normally only to get shape, but for normalized regions it also has info useful for
        # descaling the data
        self.spt_region = self.region_metadata.create_instance()
        self.logger.debug('Loaded spt_region for predictions: {} {!r}'.format(self.spt_region,
                                                                              self.spt_region))

        # set prepared flag
        self.prepared = True

    def plot_regions(self, prediction_region, output_prefix):
        '''
        Plot the partitioning, with the prediction region overlayed on top
        '''

        x_len, y_len = self.partition.shape

        # get a description for this plot, using absolute coordinates
        absolute_region = self.region_metadata.absolute_coordinates_of_region(prediction_region)
        desc_txt = 'Region: ({}, {}) - ({}, {}) - ({}, {}) - ({}, {})'
        desc = desc_txt.format(absolute_region.x1, absolute_region.y1,
                               absolute_region.x1, absolute_region.y2,
                               absolute_region.x2, absolute_region.y1,
                               absolute_region.x2, absolute_region.y2)

        # the clustering metadata should format as a nice string
        title = '{}; {}'.format(self.clustering_metadata, desc)

        # plot the clustering partition: show each cluster with its medoid (mark_points), also
        # show the rectangle corresponding to the prediction region
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        plot_util.plot_partition(self.partition, title=title, subplot=ax1,
                                 rectangle_region=prediction_region,
                                 mark_points=self.partition.medoids)

        # save figure
        fs_util.mkdir(self.metadata.output_dir(output_prefix))
        plt.draw()

        plot_name = self.get_plot_filename(prediction_region, output_prefix)
        plt.savefig(plot_name)
        self.logger.info('Saved figure: {}'.format(plot_name))

        # show figure
        plt.show()

    def get_plot_filename(self, prediction_region, output_prefix):
        '''
        Returns a string representing the filename for the plot.
        '''
        solver_plot_dir = self.metadata.output_dir(output_prefix)
        plot_filename = 'query-{!r}__region-{}-{}-{}-{}.pdf'.format(self.clustering_metadata,
                                                                    prediction_region.x1,
                                                                    prediction_region.x2,
                                                                    prediction_region.y1,
                                                                    prediction_region.y2)
        return os.path.join(solver_plot_dir, plot_filename)

    def save(self):
        '''
        Saves this solver as a pickle object for later use.
        See AutoARIMASolverPickler for details.
        '''
        # delegate to the pickler
        pickler = AutoARIMASolverPickler(solver_metadata=self.metadata,
                                         error_type=self.error_type)
        pickler.save_solver(self)

    def __str__(self):
        '''
        Solver string representation in multiple lines.
        '''
        lines = [
            'Region: {}'.format(self.region_metadata),
            'Auto ARIMA: {}'.format(self.auto_arima_params),
            '{}'.format(self.clustering_metadata),
            'Error function: {}'.format(self.error_type)
        ]
        return '\n'.join(lines)


class PredictionQueryResult(BaseRegion):
    '''
    A region that represents the result of a prediction query over a prediction region.
    For each point in the prediction region, it contains the following:

    - The forecast series: from forecast_region
    - The generalization error: from generalization_errors
    - The cluster index associated to the cluster for which the point is identified: from partition
    '''

    def __init__(self, solver_metadata, distance_measure,
                 forecast_len, forecast_subregion, error_subregion, prediction_region,
                 partition, spt_region, error_type, output_prefix):
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

        # query-specific
        self.forecast_len = forecast_len
        self.forecast_subregion = forecast_subregion
        self.error_subregion = error_subregion

        # the offset caused by changing coordinates to the prediction region
        self.prediction_region = prediction_region
        self.offset_x, self.offset_y = prediction_region.x1, prediction_region.y1

        # valid for the entire domain region
        self.partition = partition
        self.spt_region = spt_region
        self.error_type = error_type
        self.output_prefix = output_prefix

    def forecast_at(self, relative_point):
        '''
        Forecast at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.forecast_subregion.series_at(relative_point)

    def error_at(self, relative_point):
        '''
        Generalization error at point, where (0, 0) is the corner of the prediction region.
        '''
        return self.error_subregion.value_at(relative_point)

    def cluster_index_of(self, relative_point):
        '''
        Cluster that has point as member, where (0, 0) is the corner of the prediction region.
        '''
        # undo the prediction offset to get coordinates in the domain region
        domain_point = Point(relative_point.x + self.offset_x, relative_point.y + self.offset_y)

        # return the value of the mask at this point
        return self.partition.membership_of_points([domain_point])[0]

    def absolute_coordinates_of(self, relative_point):
        '''
        Computes the original coordinates of the raw dataset. This takes into consideration
        that the domain region is a subset of the complete raw dataset! Uses the region metadata.
        '''
        # undo the prediction offset to get coordinates in the domain region
        domain_point = Point(relative_point.x + self.offset_x, relative_point.y + self.offset_y)

        # now get the absolute coordinates, because the domain region can be a subset too
        return self.region_metadata.absolute_position_of_point(domain_point)

    def get_csv_file_path(self, prefix):
        '''
        Returns a string representing a CSV file for this query.
        The prefix can be used to differentiate between the CSV of the query and the CSV of the
        query summary.
        '''
        solver_csv_dir = self.metadata.output_dir(self.output_prefix)
        csv_t = '{}-{!r}__region-{}-{}-{}-{}__f{}__{}.csv'
        csv_filename = csv_t.format(prefix,
                                    self.clustering_metadata,
                                    self.prediction_region.x1,
                                    self.prediction_region.x2,
                                    self.prediction_region.y1,
                                    self.prediction_region.y2,
                                    self.forecast_len,
                                    self.error_type)

        return os.path.join(solver_csv_dir, csv_filename)

    def save_as_csv(self):
        '''
        Writes two CSV files:
        1. The prediction results (query-result-[...]), one tuple for each point in the prediction
           query, and with these columns:
            1.a. the absolute coordinates of the point
            1.b. the index of the cluster that contains the point
            1.c. generalization error at the point (see error_at())
            1.d. the forecasted series at the point (see forecast_at())

        2. The prediction summary (query-summary-[...]), a single tuple with the following info:
            2.a. clustering columns, e.g. k, seed
            2.b. number of clusters that intersect the prediction region
            2.c. MSE of the generalization errors in the prediction result

        The get_csv_file_path(prefix) method is used to calculate the file paths.
        '''

        # ensure output dir
        fs_util.mkdir(self.metadata.output_dir(self.output_prefix))

        np.set_printoptions(precision=3)

        # write the prediction results
        result_csv_file = self.get_csv_file_path('query-result')
        with open(result_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # write header
            csv_writer.writerow(['Point', 'cluster_index', 'error', 'forecast_series'])

            # write a tuple for each point in prediction region
            for relative_point in self:

                # 1.a. the absolute coordinates of the point
                coords = self.absolute_coordinates_of(relative_point)

                # 1.b. the index of the cluster that contains the point
                cluster_index = self.cluster_index_of(relative_point)

                # 1.c. generalization error at the point
                error = self.error_at(relative_point)
                error_str = '{:.3f}'.format(error)

                # 1.d. the forecasted series at the point
                forecast = self.forecast_at(relative_point)
                csv_writer.writerow([coords, cluster_index, error_str, forecast])

        self.logger.info('Wrote CSV of prediction result at: {}'.format(result_csv_file))

        # write the prediction summary
        summary_csv_file = self.get_csv_file_path('query-summary')
        with open(summary_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # write header, the clustering algorithm may have many columns
            clustering_dict = self.clustering_metadata.as_dict()
            clustering_header = clustering_dict.keys()
            csv_header = list(clustering_header)
            csv_header.extend(['clusters', 'mse'])
            csv_writer.writerow(csv_header)

            # 2.a. clustering columns, e.g. k, seed
            clustering_data = clustering_dict.values()

            # 2.b. number of clusters that intersect the prediction region
            clusters_each_point = [
                self.cluster_index_of(relative_point)
                for relative_point
                in self
            ]
            unique_clusters = list(set(clusters_each_point))
            cluster_count = len(unique_clusters)

            # 2.c. MSE of the generalization errors in the prediction result
            generalization_errors = [
                self.error_at(relative_point)
                for relative_point
                in self
            ]
            mse_error = arrays_util.mean_squared(generalization_errors)
            mse_error_str = '{:.3f}'.format(mse_error)

            # write a single row with this data
            csv_row = list(clustering_data)
            csv_row.extend([str(cluster_count), mse_error_str])
            csv_writer.writerow(csv_row)

        self.logger.info('Wrote CSV of prediction summary at: {}'.format(summary_csv_file))


class AutoARIMASolverPickler(log_util.LoggerMixin):
    '''
    Handles the persistence of a solver using pickle.
    The metadata is used to create a directory structure that uniquely identies the metadata
    parameters: different parameters should produce a different pickle file.

    Example:

    |- pickle
        |- <region>
            |- dtw
                |- kmedoids_k<k>_seed<seed>
                    |- partition.pkl
                    |- auto_arima_<auto_arima_params>_model_region.pkl
                    |- auto_arima_<auto_arima_params>_errors_<error_type>.pkl
    '''

    def __init__(self, solver_metadata, error_type):

        # solver metadata
        self.metadata = solver_metadata
        self.error_type = error_type

    def save_solver(self, auto_arima_solver):
        '''
        persist the solver details as pickle objects.
        '''
        partition = auto_arima_solver.partition
        arima_model_region = auto_arima_solver.arima_model_region
        generalization_errors = auto_arima_solver.generalization_errors
        solver_pickle_dir = self.metadata.pickle_dir()

        # create the directory
        fs_util.mkdir(solver_pickle_dir)

        # save the partition
        partition_path = self.partition_pickle_path()
        with open(partition_path, 'wb') as pickle_file:
            pickle.dump(partition, pickle_file)
            self.logger.debug('Saved partition at {}'.format(partition_path))

        # save the arima model region
        arima_model_path = self.arima_model_pickle_path()
        with open(arima_model_path, 'wb') as pickle_file:
            pickle.dump(arima_model_region, pickle_file)
            self.logger.debug('Saved arima_model region at {}'.format(arima_model_path))

        # save the generalization errors
        generalization_errors_path = self.generalization_errors_pickle_path()
        with open(generalization_errors_path, 'wb') as pickle_file:
            pickle.dump(generalization_errors, pickle_file)
            self.logger.debug('Saved errors at {}'.format(generalization_errors_path))

        self.logger.info('Solver saved at {}'.format(solver_pickle_dir))

    def load_solver(self):
        '''
        Restores a solver from pickled objects. The inverse operation of save_solver.
        Notice that this operation requires the metadata.
        '''

        # load the cluster partition
        partition_path = self.partition_pickle_path()
        self.logger.debug('Attempting to load cluster partition at {}'.format(partition_path))
        with open(partition_path, 'rb') as pickle_file:
            partition = pickle.load(pickle_file)

        # load the arima model region
        arima_model_path = self.arima_model_pickle_path()
        self.logger.debug('Attempting to load arima model region at {}'.format(arima_model_path))
        with open(arima_model_path, 'rb') as pickle_file:
            arima_model_region = pickle.load(pickle_file)

        # load the generalization errors
        generalization_errors_path = self.generalization_errors_pickle_path()
        self.logger.debug('Attempting to errors at {}'.format(generalization_errors_path))
        with open(generalization_errors_path, 'rb') as pickle_file:
            generalization_errors = pickle.load(pickle_file)

        # recreate the solver
        log_msg = 'Loaded solver: {}, {}, {}, {}'
        self.logger.info(log_msg.format(self.metadata.region_metadata,
                                        self.metadata.clustering_metadata,
                                        self.metadata.model_params,
                                        self.error_type))

        return AutoARIMASolver(solver_metadata=self.metadata,
                               error_type=self.error_type,
                               partition=partition,
                               arima_model_region=arima_model_region,
                               generalization_errors=generalization_errors)

    def partition_pickle_path(self):
        '''
        Full path to pickle object of the partition, given the region, clustering and distance
        metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        return os.path.join(solver_pickle_dir, 'partition.pkl')

    def arima_model_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region, given by the region, clustering,
        distance and auto arima metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        arima_model_filename = 'model-region__{!r}.pkl'.format(self.metadata.model_params)
        return os.path.join(solver_pickle_dir, arima_model_filename)

    def generalization_errors_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region, given by the region, clustering,
        distance and auto arima metadata, also the error type.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        generalization_errors_filename = 'errors__{!r}__{}.pkl'.format(self.metadata.model_params,
                                                                       self.error_type)
        return os.path.join(solver_pickle_dir, generalization_errors_filename)
