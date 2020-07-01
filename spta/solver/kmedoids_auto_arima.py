'''
Produces temporal forecasts for points in the region, based on the model trained at medoids.
Uses kmedoids to define clusters and medoids, and uses auto ARIMA as the forecast model.
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import csv
import numpy as np
import pickle
import os

from spta.region import Point
from spta.region.base import BaseRegion
from spta.region.error import MeasureForecastingError, get_error_func
from spta.region.partition import PartitionRegionCrisp
from spta.region.spatial import SpatialCluster
from spta.region.temporal import SpatioTemporalRegion
from spta.region.train import SplitTrainingAndTestLast

from spta.arima.forecast import ArimaModelRegion
from spta.arima import training

from spta.kmedoids import kmedoids

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util


# default forecast length
FORECAST_LENGTH = 8


class KmedoidsAutoARIMATrainer(log_util.LoggerMixin):
    '''
    Trains the ARIMA models by partitioning the region and using the ARIMA models at the medoids
    of the resulting clusters
    '''

    def __init__(self, region_metadata, kmedoids_metadata, auto_arima_params):
        self.region_metadata = region_metadata
        self.auto_arima_params = auto_arima_params

        # use pre-computed distance matrix
        # TODO this code is broken if we don't use DTW
        distance_measure = kmedoids_metadata.distance_measure
        distance_measure.load_distance_matrix_2d(self.region_metadata.distances_filename,
                                                 self.region_metadata.region)

        self.kmedoids_metadata = kmedoids_metadata
        self.distance_measure = distance_measure

    def train(self, error_type, training_len=None, test_len=FORECAST_LENGTH):
        '''
        Partition the region into clusters with kmedoids, then train the ARIMA models on the
        cluster medoids.

        This code assumes that each temporal series is split into training and test subseries.
        Needs to change when we use a fixed number of points in the past to train the model.
        '''

        # get the region and transform to list of time series
        spt_region = SpatioTemporalRegion.from_metadata(self.region_metadata)

        # run k-medoids and get the result
        series_group = spt_region.as_2d
        kmedoids_result = kmedoids.run_kmedoids_from_metadata(series_group, self.kmedoids_metadata)

        # create training/test regions
        splitter = SplitTrainingAndTestLast(test_len)
        (training_region, test_region) = splitter.split(spt_region)

        # use the k-medoids result to train ARIMA models at the medoids and use them as
        # representatives for their own clusters. These models will be used to evalute and save
        # the prediction, but will not be the final models.
        # The output is a single spatial region with k different models!
        arima_model_region_for_error = self.train_auto_arima_at_medoids(training_region,
                                                                        kmedoids_result)

        # calculate the generalization error for these ARIMA models: it is the prediction error
        # that is calculated using the test dataset.
        # TODO a lot of assumptions here, work them out later
        generalization_errors = self.calculate_errors(arima_model_region_for_error,
                                                      training_region, test_region,
                                                      forecast_len=test_len,
                                                      error_type=error_type)

        # we have the errors, these will be saved
        # now we need to re-train the models at the medoids with the full dataset
        # which will be used for prediction
        arima_model_region_for_predict = self.train_auto_arima_at_medoids(spt_region,
                                                                          kmedoids_result)

        # create a solver with the data acquired, this solver can answer queries
        solver = KmedoidsAutoARIMASolver(self.region_metadata,
                                         self.kmedoids_metadata,
                                         self.auto_arima_params,
                                         error_type,
                                         kmedoids_result,
                                         arima_model_region_for_predict,
                                         generalization_errors)
        return solver

    def train_auto_arima_at_medoids(self, training_region, kmedoids_result):
        '''
        The ARIMA model region that will be used for forecasting has only k models; the model from
        each medoid will be replicated throughout its cluster.
        This is achieved by creating k clusters, each containing the trained model at its
        medoid, then merging the clusters with the values at the representatives.

        See PartitionRegionCrisp.merge_with_representatives_2d for details.
        '''
        _, x_len, y_len = training_region.shape

        # get the medoids as points
        medoid_indices = kmedoids_result.medoids
        medoid_points = [
            # get (i, j) position relative to the region
            Point(int(medoid_index / y_len), medoid_index % y_len)
            for medoid_index
            in medoid_indices
        ]

        # use k-medoids result to create clusters for the training dataset
        # these clusters have medoids from k-medoids as the centroids
        # the idea is to use train ARIMA models at these medoids
        membership = kmedoids_result.labels
        partition = PartitionRegionCrisp.from_membership_array(membership, x_len, y_len)
        training_clusters = partition.create_all_spt_clusters(training_region,
                                                              centroid_indices=medoid_indices)

        # the 'sparse' numpy array that will store the ARIMA models and has the region shape
        # note that all clusters share the same underlying dataset (should be no problem...)
        arima_medoids_numpy = np.empty((x_len, y_len), dtype=object)

        # iterate the training clusters to produce an ARIMA cluster for each one
        # using enumerate to also get the cluster index in [0, k-1]
        arima_clusters = []
        for cluster_index, training_cluster in enumerate(training_clusters):

            arima_cluster = self.train_auto_arima_at_medoid(partition, training_cluster,
                                                            arima_medoids_numpy, cluster_index)
            arima_clusters.append(arima_cluster)

        # merge the clusters into a single region, this will be backed by a new numpy matrix
        # but should look exactly like arima_medoids_numpy
        arima_region = partition.merge_with_representatives_2d(arima_clusters, medoid_points)
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

    def train_auto_arima_at_medoid(self, partition, training_cluster, arima_medoids_numpy,
                                   cluster_index):
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
                                             cluster_index=cluster_index)
        return arima_model_cluster


class KmedoidsAutoARIMASolver(log_util.LoggerMixin):
    '''
    Represents a cluster partition of a given spatio-temporal region, with ARIMA models trained
    at the medoids. It can answer a forecast query of a specified subregion, by using, at each
    point in the subregion, the forecast at the medoid of the cluster for which each point is
    a member.
    '''
    def __init__(self, region_metadata, kmedoids_metadata, auto_arima_params, error_type,
                 kmedoids_result, arima_model_region, generalization_errors):

        # user input
        self.region_metadata = region_metadata
        self.kmedoids_metadata = kmedoids_metadata
        self.auto_arima_params = auto_arima_params
        self.error_type = error_type

        # calculated during training
        self.kmedoids_result = kmedoids_result
        self.arima_model_region = arima_model_region
        self.generalization_errors = generalization_errors

        # flag
        self.prepared = False

    def predict(self, prediction_region, forecast_len=FORECAST_LENGTH, plot=True):
        self.logger.debug('Predicting for region: {}'.format(prediction_region))

        if not self.prepared:
            self.prepare_for_predictions()

        # plot the whole region and the prediction region
        self.plot_regions(prediction_region)

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

        # this has all the required information and can be iterated
        return PredictionQueryResult(forecast_len, forecast_subregion, error_subregion,
                                     prediction_region, self.partition, self.spt_region,
                                     self.kmedoids_metadata, self.error_type)

    def prepare_for_predictions(self):
        '''
        Prepare to attend prediction requests. For now, this is called on-demand.
        In a distant future, it can be used to lazily instantiate a solver.
        '''

        # load the region again
        # normally only to get shape, but for normalized regions it also has info useful for
        # denormalizing the data
        self.spt_region = SpatioTemporalRegion.from_metadata(self.region_metadata)
        _, x_len, y_len = self.spt_region.shape

        # rebuild the cluster partition
        membership = self.kmedoids_result.labels
        self.partition = PartitionRegionCrisp.from_membership_array(membership, x_len, y_len)

        # set prepared flag
        self.prepared = True

    def plot_regions(self, prediction_region):
        '''
        Plot the partitioning, with the prediction region overlayed on top
        '''

        x_len, y_len = self.partition.shape

        # 1-d labels required for the plot
        membership_1d = self.partition.as_numpy.reshape(x_len * y_len)
        shape_2d = (x_len, y_len)

        # plot the partitioning
        title = 'Kmedoids: k={} seed={}; Query: {}'.format(self.kmedoids_result.k,
                                                           self.kmedoids_result.random_seed,
                                                           prediction_region)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        plot_util.plot_2d_clusters(membership_1d, shape_2d, title=title, subplot=ax1)

        # add the rectangle representing the region
        width = prediction_region.x2 - prediction_region.x1
        height = prediction_region.y2 - prediction_region.y1

        # https://stackoverflow.com/questions/52056475/python-plot-rectangles-of-known-size-at-scatter-points
        ax1.add_patch(Rectangle(xy=(prediction_region.x1 - 0.5, prediction_region.y1 - 0.5),
                                width=width, height=height, linewidth=1, color='red', fill=False))

        # save figure
        fs_util.mkdir(self.get_plot_directory_structure())
        plt.draw()

        plot_name = self.get_plot_filename(prediction_region)
        plt.savefig(plot_name)
        self.logger.info('Saved figure: {}'.format(plot_name))

        # show figure
        plt.show()

    def get_plot_directory_structure(self):
        '''
        Returns a string representing the directory structure for storing plots.
        Ex:
            plots/nordeste_small_1y_1ppd_last/kmedoids/k8_seed1_dtw/
        '''

        directory_str = 'plots/{}/kmedoids/k{}_seed{}_{}'
        return directory_str.format(self.region_metadata,
                                    self.kmedoids_metadata.k,
                                    self.kmedoids_metadata.random_seed,
                                    self.kmedoids_metadata.distance_measure)

    def get_plot_filename(self, prediction_region):
        '''
        Returns a string representing the filename for the plot.
        Ex:
            csv/nordeste_small_1y_1ppd_last/kmedoids/k8_seed1_dtw/
            region_1_4_2_4.pdf
        '''
        solver_plot_dir = self.get_plot_directory_structure()
        plot_filename = 'region_{}-{}-{}-{}.pdf'.format(prediction_region.x1,
                                                        prediction_region.x2,
                                                        prediction_region.y1,
                                                        prediction_region.y2)
        return os.path.join(solver_plot_dir, plot_filename)

    def save(self):
        '''
        Saves this solver as a pickle object for later use.
        See KmedoidsAutoARIMASolverPickler for details.
        '''
        # delegate to the pickler
        pickler = KmedoidsAutoARIMASolverPickler(self.region_metadata,
                                                 self.kmedoids_metadata,
                                                 self.auto_arima_params,
                                                 self.error_type)
        pickler.save_solver(self)

    def __str__(self):
        '''
        Solver string representation in multiple lines.
        '''
        lines = [
            'Region: {}'.format(self.region_metadata),
            'Auto ARIMA: {}'.format(self.auto_arima_params),
            'K-mediods: k={}, seed={}'.format(self.kmedoids_metadata.k,
                                              self.kmedoids_metadata.random_seed),
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

    def __init__(self, forecast_len, forecast_subregion, error_subregion, prediction_region,
                 partition, spt_region, kmedoids_metadata, error_type):
        '''
        Constructor, uses a subset of the numpy matrix that backs the forecast region, the subset
        is taken using the prediction region coordinates.
        '''
        super(PredictionQueryResult, self).__init__(forecast_subregion.as_numpy)

        self.forecast_len = forecast_len
        self.forecast_subregion = forecast_subregion
        self.error_subregion = error_subregion

        # the offset caused by changing coordinates to the prediction region
        self.prediction_region = prediction_region
        self.offset_x, self.offset_y = prediction_region.x1, prediction_region.y1

        # note that these are not changed! valid for the entire domain region
        self.partition = partition
        self.spt_region = spt_region
        self.region_metadata = spt_region.region_metadata
        self.kmedoids_metadata = kmedoids_metadata
        self.error_type = error_type

    def forecast_at(self, relative_point):
        '''
        Forecast at point, where (0, 0) is the corner of the prediction region.
        We need to handle denormalization here!
        '''
        forecast_series = self.forecast_subregion.series_at(relative_point)

        if self.region_metadata.normalized:

            # undo the prediction offset to get coordinates in the domain region
            domain_point = Point(relative_point.x + self.offset_x,
                                 relative_point.y + self.offset_y)

            # get the normalization info for this domain point
            norm_min = self.spt_region.normalization_min.value_at(domain_point)
            norm_max = self.spt_region.normalization_max.value_at(domain_point)
            self.logger.debug('point {}: norm_min={}, norm_max={}'.format(domain_point,
                                                                          norm_min, norm_max))

            # denormalize
            forecast_series = (norm_max - norm_min) * forecast_series + norm_min

        return forecast_series

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

    def get_directory_structure(self):
        '''
        Returns a string representing the directory structure for storing query results.
        Ex:
            csv/nordeste_small_1y_1ppd_last/kmedoids/k8_seed1_dtw/
        '''

        directory_str = 'csv/{}/kmedoids/k{}_seed{}_{}'
        return directory_str.format(self.region_metadata,
                                    self.kmedoids_metadata.k,
                                    self.kmedoids_metadata.random_seed,
                                    self.kmedoids_metadata.distance_measure)

    def get_csv_file_path(self):
        '''
        Returns a string representing the CSV file for this query.
        Ex:
            csv//kmedoids/k8_seed1_dtw/
            region_1_4_2_4_forecast_8_sMAPE.csv
        '''
        solver_csv_dir = self.get_directory_structure()
        csv_filename = 'region_{}-{}-{}-{}_forecast_{}_{}.csv'.format(self.prediction_region.x1,
                                                                      self.prediction_region.x2,
                                                                      self.prediction_region.y1,
                                                                      self.prediction_region.y2,
                                                                      self.forecast_len,
                                                                      self.error_type)
        return os.path.join(solver_csv_dir, csv_filename)

    def save_as_csv(self):
        '''
        Writes prediction results to CSV, at the file indicated by get_csv_file_path()
        '''

        # essure dir
        solver_csv_dir = self.get_directory_structure()
        fs_util.mkdir(solver_csv_dir)

        np.set_printoptions(precision=3)
        csv_filename = self.get_csv_file_path()

        with open(csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            # write header
            csv_writer.writerow(['Point', 'cluster_index', 'error', 'forecast_series'])

            # write a tuple for each point in prediction region
            for relative_point in self:
                coords = self.absolute_coordinates_of(relative_point)
                cluster_index = self.cluster_index_of(relative_point)

                error = self.error_at(relative_point)
                error_str = '{:.3f}'.format(error)

                forecast = self.forecast_at(relative_point)
                csv_writer.writerow([coords, cluster_index, error_str, forecast])

        self.logger.info('Wrote CSV result at: {}'.format(csv_filename))


class KmedoidsAutoARIMASolverPickler(log_util.LoggerMixin):
    '''
    Handles the persistence of a solver using pickle.
    The metadata is used to create a directory structure that uniquely identies the metadata
    parameters: different parameters should produce a different pickle file.

    |- pickle
        |- <region>
            |- kmedoids
                |-  k<k>_seed<seed>_dtw
                    |- kmedoids_result.pkl
                    |- auto_arima_<auto_arima_params>_model_region.pkl
                    |- auto_arima_<auto_arima_params>_errors_<error_type>.pkl
    '''

    def __init__(self, region_metadata, kmedoids_metadata, auto_arima_params, error_type):
        self.region_metadata = region_metadata
        self.kmedoids_metadata = kmedoids_metadata
        self.auto_arima_params = auto_arima_params
        self.error_type = error_type

    def save_solver(self, kmedoids_auto_arima_solver):
        '''
        persist the solver details as pickle objects.
        '''
        kmedoids_result = kmedoids_auto_arima_solver.kmedoids_result
        arima_model_region = kmedoids_auto_arima_solver.arima_model_region
        generalization_errors = kmedoids_auto_arima_solver.generalization_errors
        solver_dir = self.get_directory_structure()

        # create the directory
        fs_util.mkdir(solver_dir)

        # save the kmedoids result
        kmedoids_result_path = self.kmedoids_result_pickle_path()
        with open(kmedoids_result_path, 'wb') as pickle_file:
            pickle.dump(kmedoids_result, pickle_file)
            self.logger.debug('Saved kmedoids_result at {}'.format(kmedoids_result_path))

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

        self.logger.info('Solver saved at {}'.format(solver_dir))

    def load_solver(self):
        '''
        Restores a solver from pickled objects. The inverse operation of save_solver.
        Notice that this operation requires the metadata.
        '''

        # load the kmedoids result
        kmedoids_result_path = self.kmedoids_result_pickle_path()
        self.logger.debug('Attempting to load kmedoids result at {}'.format(kmedoids_result_path))
        with open(kmedoids_result_path, 'rb') as pickle_file:
            kmedoids_result = pickle.load(pickle_file)

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
        log_msg = 'Loaded solver: {}, k={}, seed={}, {}, {}'
        self.logger.info(log_msg.format(self.region_metadata,
                                        kmedoids_result.k, kmedoids_result.random_seed,
                                        self.auto_arima_params, self.error_type))

        return KmedoidsAutoARIMASolver(region_metadata=self.region_metadata,
                                       kmedoids_metadata=self.kmedoids_metadata,
                                       auto_arima_params=self.auto_arima_params,
                                       error_type=self.error_type,
                                       kmedoids_result=kmedoids_result,
                                       arima_model_region=arima_model_region,
                                       generalization_errors=generalization_errors)

    def kmedoids_result_pickle_path(self):
        '''
        Full path to pickle object of the kmedoids result, given the region and kmedoids metadata.
        '''
        solver_dir = self.get_directory_structure()
        kmedoids_filename = self.kmedoids_result_filename()
        return os.path.join(solver_dir, kmedoids_filename)

    def arima_model_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region, given by the region, kmedoids
        and auto arima metadata.
        '''
        solver_dir = self.get_directory_structure()
        arima_model_filename = self.auto_arima_model_filename()
        return os.path.join(solver_dir, arima_model_filename)

    def generalization_errors_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region, given by the region, kmedoids
        and auto arima metadata, and the error type.
        '''
        solver_dir = self.get_directory_structure()
        generalization_errors_filename = self.generalization_errors_filename()
        return os.path.join(solver_dir, generalization_errors_filename)

    def get_directory_structure(self):
        '''
        Returns a string representing the directory structure for storing solver data.
        Ex:
            pickle/nordeste_small_1y_1ppd_last/kmedoids/k8_seed1_dtw/
        '''

        directory_str = 'pickle/{}/kmedoids/k{}_seed{}_{}'
        return directory_str.format(self.region_metadata,
                                    self.kmedoids_metadata.k,
                                    self.kmedoids_metadata.random_seed,
                                    self.kmedoids_metadata.distance_measure)

    def kmedoids_result_filename(self):
        return 'kmedoids_result.pkl'

    def auto_arima_model_filename(self):
        '''
        auto_arima_<auto_arima_params>_model_region.pkl
        See spta.arima.AutoArimaParams
        '''
        return 'auto_arima_{}_model_region.pkl'.format(self.auto_arima_params)

    def generalization_errors_filename(self):
        '''
        auto_arima_<auto_arima_params>_errors_<error_type>.pkl
        '''
        return 'auto_arima_{}_errors_{}.pkl'.format(self.auto_arima_params, self.error_type)
