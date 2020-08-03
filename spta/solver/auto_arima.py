'''
Produces temporal forecasts for points in the region, based on the model trained at medoids.
Uses a clustering algorithm to define clusters and medoids, and uses auto ARIMA for forecasting.
'''
import matplotlib.pyplot as plt

import numpy as np
import pickle
import os

from spta.arima.forecast import ArimaModelRegion
from spta.arima import training

from spta.clustering.factory import ClusteringFactory

from spta.region.error import MeasureForecastingError, get_error_func
from spta.region.scaling import SpatioTemporalScaled
from spta.region.spatial import SpatialCluster
from spta.region.train import SplitTrainingAndTestLast

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from .metadata import SolverMetadata
from .result import PredictionQueryResult

# default forecast length
FORECAST_LENGTH = 8


class AutoARIMATrainer(log_util.LoggerMixin):
    '''
    Trains the ARIMA models by partitioning the region and using the ARIMA models at the medoids
    of the resulting clusters
    '''

    def __init__(self, region_metadata, clustering_metadata, distance_measure, auto_arima_params,
                 error_type):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.auto_arima_params = auto_arima_params
        self.error_type = error_type

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
                                       model_params=self.auto_arima_params,
                                       error_type=self.error_type)
        self.prepared = True

    def train(self, training_len=None, test_len=FORECAST_LENGTH, output_prefix='outputs'):
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

        # get the cluster partition and corresponding medoids, save the CSV and pickle
        # when the partition is saved, the medoids are saved in it
        partition = self.clustering_algorithm.partition(spt_region,
                                                        with_medoids=True,
                                                        save_csv_at=output_prefix,
                                                        pickle_prefix='pickle')
        medoids = partition.medoids

        self.logger.info('Training solver: {}'.format(self.metadata))

        # use the partition to train ARIMA models at the cluster medoids and use them as
        # representatives for their own clusters. These models will be used to evalute and save
        # the generalization error. They can also be used for prediction, unless the future flag
        # is set (see below).
        # The output is a single spatial region with k different models!
        arima_model_region_training = self.train_auto_arima_at_medoids(training_region, partition,
                                                                       medoids)

        # calculate the generalization error for these ARIMA models: it is the prediction error
        # that is calculated using the test dataset.
        # TODO a lot of assumptions here, work them out later
        generalization_errors = self.calculate_errors(arima_model_region_training,
                                                      training_region, test_region,
                                                      forecast_len=test_len)
        self.logger.info('Overall error: {:.4f}'.format(generalization_errors.overall_error))

        # If the user sets the future flag for prediction, we want to use the entire series
        # to make out-of-sample predictions (out of sample)
        # For that, we need to re-train the models at the medoids again, these time using the
        # full dataset (not the training subset, which uses a series subset)
        arima_model_region_whole = self.train_auto_arima_at_medoids(spt_region, partition,
                                                                    medoids)

        # create a solver with the data acquired, this solver can answer queries
        solver = AutoARIMASolver(solver_metadata=self.metadata,
                                 partition=partition,
                                 arima_model_region_training=arima_model_region_training,
                                 arima_model_region_whole=arima_model_region_whole,
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

    def calculate_errors(self, arima_model_region, training_region, test_region, forecast_len):
        '''
        Calculate the generalization error for these ARIMA models: it is the prediction error
        that is calculated using the test dataset.

        TODO a lot of assumptions here, work them out later
        '''
        # create a forecast: ARIMA requires an empty region (no data is required, only shape)
        # use the test region for convenience
        forecast_region = arima_model_region.apply_to(test_region, forecast_len)

        # calculate the error using a generic error function, requires the error type
        error_func = get_error_func(self.error_type)
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
    def __init__(self, solver_metadata, partition, arima_model_region_training,
                 arima_model_region_whole, generalization_errors):

        # user input
        self.metadata = solver_metadata
        self.region_metadata = solver_metadata.region_metadata
        self.clustering_metadata = solver_metadata.clustering_metadata
        self.distance_measure = solver_metadata.distance_measure
        self.auto_arima_params = solver_metadata.model_params
        self.error_type = solver_metadata.error_type

        # calculated during training
        self.partition = partition
        self.arima_model_region_training = arima_model_region_training
        self.arima_model_region_whole = arima_model_region_whole
        self.generalization_errors = generalization_errors

        # flag
        self.prepared = False

    def predict(self, prediction_region, is_future, forecast_len=FORECAST_LENGTH,
                output_prefix='outputs', plot=True):

        self.logger.debug('Predicting for region: {}'.format(prediction_region))

        if not self.prepared:
            self.prepare_for_predictions(forecast_len)

        # plot the whole region and the prediction region
        self.plot_regions(prediction_region, output_prefix)

        px1, px2, py1, py2 = prediction_region.x1, prediction_region.x2, prediction_region.y1, \
            prediction_region.y2

        # the generalization errors are precalculated, just subset them
        error_subregion = self.generalization_errors.region_subset(prediction_region)

        # in-sample (is_future False) or out-of-sample (is_future True)?
        if is_future:
            # this ARIMA model was trained with the whole dataset, so it will produce out-of-sample
            # predictions
            arima_model_region = self.arima_model_region_whole

            # no test data available for out-of-sample
            test_subregion = None

        else:
            # this ARIMA model was trained with the training dataset, so it will produce in-sample
            # predictions (comparable to the test)
            arima_model_region = self.arima_model_region_training

            # extract the test data for the prediction region
            test_subregion = self.test_region.region_subset(prediction_region)

        # compute the prediction: create a forecast over the region
        # to do this, obtain a subset of the model region
        # TODO better support for this...
        arima_models_subset_np = arima_model_region.as_numpy[px1:px2, py1:py2]
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
                                     is_future=is_future,
                                     forecast_subregion=forecast_subregion,
                                     test_subregion=test_subregion,
                                     error_subregion=error_subregion,
                                     prediction_region=prediction_region,
                                     partition=self.partition,
                                     spt_region=self.spt_region,
                                     output_prefix=output_prefix)

    def prepare_for_predictions(self, forecast_len):
        '''
        Prepare to attend prediction requests. For now, this is called on-demand.
        In a distant future, it can be used to lazily instantiate a solver.

        forecast_len: the length of the forecasted series, here used to get the test series
        '''

        # load the region again
        # normally only to get shape, but for normalized regions it also has info useful for
        # descaling the data
        self.spt_region = self.region_metadata.create_instance()
        self.logger.debug('Loaded spt_region for predictions: {} {!r}'.format(self.spt_region,
                                                                              self.spt_region))

        # get test region, apply scaling if needed
        splitter = SplitTrainingAndTestLast(forecast_len)
        (_, self.test_region) = splitter.split(self.spt_region)
        if self.test_region.has_scaling():
            self.test_region = self.test_region.descale()

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
        pickler = AutoARIMASolverPickler(solver_metadata=self.metadata)
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

    def __init__(self, solver_metadata):

        # solver metadata
        self.metadata = solver_metadata

        self.distance_measure = self.metadata.distance_measure
        self.clustering_metadata = self.metadata.clustering_metadata
        self.region_metadata = self.metadata.region_metadata
        self.error_type = self.metadata.error_type

    def save_solver(self, auto_arima_solver):
        '''
        persist the solver details as pickle objects.
        '''
        arima_model_region_training = auto_arima_solver.arima_model_region_training
        arima_model_region_whole = auto_arima_solver.arima_model_region_whole
        generalization_errors = auto_arima_solver.generalization_errors

        # create the directory
        solver_pickle_dir = self.metadata.pickle_dir()
        fs_util.mkdir(solver_pickle_dir)

        # the clustering algorithm is already managing partition persistence when it creates it

        # save the arima model region based on training dataset
        arima_model_training_path = self.arima_model_training_pickle_path()
        with open(arima_model_training_path, 'wb') as pickle_file:
            pickle.dump(arima_model_region_training, pickle_file)
            self.logger.debug('Saved arima_model (training): {}'.format(arima_model_training_path))

        # save the arima model region for whole dataset
        arima_model_whole_path = self.arima_model_whole_pickle_path()
        with open(arima_model_whole_path, 'wb') as pickle_file:
            pickle.dump(arima_model_region_whole, pickle_file)
            self.logger.debug('Saved arima_model (whole): {}'.format(arima_model_whole_path))

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
        # the clustering algorithm can manage partition persistence,
        # but need to create an instance of this algorithm to delegate task
        # TODO get a better handle for the functionality?
        clustering_factory = ClusteringFactory(self.distance_measure)
        clustering_algorithm = clustering_factory.instance(self.clustering_metadata)
        partition = clustering_algorithm.load_previous_partition(self.region_metadata,
                                                                 pickle_prefix='pickle')

        # load the arima model region based on training data
        arima_model_training_path = self.arima_model_training_pickle_path()
        self.logger.debug('Loading arima models (training): {}'.format(arima_model_training_path))
        with open(arima_model_training_path, 'rb') as pickle_file:
            arima_model_region_training = pickle.load(pickle_file)

        # load the arima model region based on whole data
        arima_model_whole_path = self.arima_model_whole_pickle_path()
        self.logger.debug('Loading arima models (whole): {}'.format(arima_model_whole_path))
        with open(arima_model_whole_path, 'rb') as pickle_file:
            arima_model_region_whole = pickle.load(pickle_file)

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
                               partition=partition,
                               arima_model_region_training=arima_model_region_training,
                               arima_model_region_whole=arima_model_region_whole,
                               generalization_errors=generalization_errors)

    def partition_pickle_path(self):
        '''
        Full path to pickle object of the partition, given the region, clustering and distance
        metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        return os.path.join(solver_pickle_dir, 'partition.pkl')

    def arima_model_training_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region based on training dataset; given
        by the region, clustering, distance and auto arima metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        arima_model_filename = 'model-region-training__{!r}.pkl'.format(self.metadata.model_params)
        return os.path.join(solver_pickle_dir, arima_model_filename)

    def arima_model_whole_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region based on the whole dataset
        (as opposed to just the training dataset); given by the region, clustering, distance and
        auto arima metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        arima_model_filename = 'model-region-whole__{!r}.pkl'.format(self.metadata.model_params)
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
