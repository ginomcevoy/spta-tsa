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
from spta.region.train import SplitTrainingAndTestLast

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from .metadata import SolverMetadataBuilder
from .result import PredictionQueryResultBuilder


class AutoARIMATrainer(log_util.LoggerMixin):
    '''
    Trains the ARIMA models by partitioning the region and using the ARIMA models at the medoids
    of the resulting clusters
    '''

    def __init__(self, region_metadata, clustering_metadata, distance_measure, auto_arima_params,
                 test_len, error_type):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.auto_arima_params = auto_arima_params
        self.test_len = test_len
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

        # create metadata with clustering support
        builder = SolverMetadataBuilder(region_metadata=self.region_metadata,
                                        model_params=self.auto_arima_params,
                                        test_len=self.test_len,
                                        error_type=self.error_type)
        self.metadata = builder.with_clustering(clustering_metadata=self.clustering_metadata,
                                                distance_measure=self.distance_measure).build()

        self.prepared = True

    def train(self, output_home='outputs'):
        '''
        Partition the region into clusters, then train the ARIMA models on the cluster medoids.

        The models are trained using the training samples, and the test samples (out-of-time for
        the trained models) are used to calculated the forecast error. After this is done, the
        models are re-fitted using the full dataset, in order to improve out-of-sample forecast,
        called the 'future' scenario.

        For the refit, the original hyper-parameters are kept, instead of calculating a completely
        new model with auto ARIMA.

        NOTE: This code assumes that each temporal series is split into training and test
        subseries, where training_len = series_len - test_len. This assumption will be reconsidered
        if a new requirement to specify number of training samples appears.
        '''
        if not self.prepared:
            self.prepare_for_training()

        # recover the spatio-temporal region
        spt_region = self.region_metadata.create_instance()

        # create training/test regions
        splitter = SplitTrainingAndTestLast(self.test_len)
        (training_region, test_region) = splitter.split(spt_region)

        # get the cluster partition and corresponding medoids, save the CSV and pickle
        # when the partition is saved, the medoids are saved in it
        partition = self.clustering_algorithm.partition(spt_region,
                                                        with_medoids=True,
                                                        save_csv_at=output_home,
                                                        pickle_home='pickle')
        medoids = partition.medoids

        self.logger.info('Training solver: {}'.format(self.metadata))

        # The ARIMA model region that will be used for forecasting has only k models;
        # the model from each medoid will be replicated throughout its cluster.
        #
        # This is achieved in several steps:
        #  1. Train auto ARIMA models at each medoid, and save the results in an ARIMA model region
        #     backed by sparse numpy dataset that has no models (None) outside the medoids.
        #     This step is achieved by train_auto_arima_at_medoids().
        #
        #  2. Use the partition to create k clusters, each containing the trained model at its
        #     medoid, then merging the clusters with the values at the representatives.
        #     This step is achieved by replicate_representative_models().
        #

        # do 1. here
        arima_medoid_models_train = self.train_auto_arima_at_medoids(training_region, medoids)

        # do 2. here
        arima_replicated_models_train = \
            self.replicate_representative_models(arima_medoid_models_train, partition, medoids)

        # calculate the generalization error for these ARIMA models: it is the prediction error
        # that is calculated using the test dataset.
        generalization_errors = self.calculate_errors(arima_replicated_models_train,
                                                      training_region, test_region)
        self.logger.info('Overall error: {:.4f}'.format(generalization_errors.overall_error))

        # If the user sets the future flag for prediction, we want to use the entire series
        # to make out-of-sample predictions (out of sample)
        # For that, we need to re-train the models at the medoids again, these time using the
        # full dataset (not the training subset, which uses a series subset)
        # Note that we must use arima_medoid_models_train instead of arima_replicated_models_train,
        # otherwise we would be re-training x_len * y_len models instead of just k!
        arima_medoid_models_whole = training.refit_arima(arima_medoid_models_train,
                                                         spt_region)

        # replicate these refitted models, similar to step 2. above
        arima_replicated_models_whole = \
            self.replicate_representative_models(arima_medoid_models_whole, partition, medoids)

        # create a solver with the data acquired, this solver can answer queries
        solver = AutoARIMASolver(solver_metadata=self.metadata,
                                 partition=partition,
                                 arima_model_region_training=arima_replicated_models_train,
                                 arima_model_region_whole=arima_replicated_models_whole,
                                 generalization_errors=generalization_errors)
        return solver

    def train_auto_arima_at_medoids(self, training_region, medoids):
        '''
        The ARIMA model region that will be used for forecasting has only k models; the model from
        each medoid will be replicated throughout its cluster.

        Here, wet rain auto ARIMA models at each medoid, and save the results in an ARIMA model
        region backed by sparse numpy dataset that has no models (None) outside the medoids.
        '''
        _, x_len, y_len = training_region.shape

        # The sparse numpy array that will store the ARIMA models and has the region shape.
        # The model at points other than the medoids will be None.
        arima_medoids_numpy = np.full((x_len, y_len), None, dtype=object)

        for medoid in medoids:

            # train an ARIMA model and store it at the medoid coordinates
            training_series_at_medoid = training_region.series_at(medoid)
            arima_at_medoid = training.train_auto_arima(self.auto_arima_params,
                                                        training_series_at_medoid)
            arima_medoids_numpy[medoid.x, medoid.y] = arima_at_medoid

        # wrap the dataset into a model region, which is also a spatial region
        return ArimaModelRegion(arima_medoids_numpy)

    def replicate_representative_models(self, arima_models_at_medoids, partition, medoids):
        '''
        Given a sparse ARIMA model region, replicate the models at each medoid over all the points
        of their respective clusters.
        '''
        # the clusters determine how to replicate the k ARIMA models
        arima_clusters = partition.create_all_spatial_clusters(arima_models_at_medoids)

        # Merge the clusters into a single region by replicating the models at the medoids.
        # Note that this is just a spatial region
        arima_spatial_region = partition.merge_with_representatives_2d(arima_clusters, medoids)

        # We want an ArimaModelRegion that can be applied to produce forecasts, so we wrap
        # the data with this instance
        return ArimaModelRegion(arima_spatial_region.as_numpy)

    def calculate_errors(self, arima_model_region, training_region, test_region):
        '''
        Calculate the generalization error for these ARIMA models: it is the prediction error
        that is calculated using the test dataset. For these ARIMA models, the test data is
        out-of-sample, because they were not trained with it.

        The number of forecast samples is fixed to be equal to the length of the test series,
        determined by the user request.
        '''
        # forecast the same number of samples for which we have test data
        forecast_len = test_region.series_len

        # create a forecast: ARIMA requires an empty region (no data is required, only shape)
        # use the test region for convenience
        forecast_region = arima_model_region.apply_to(test_region, forecast_len)

        # calculate the error using a generic error function, requires the error type
        error_func = get_error_func(self.error_type)
        measure_error = MeasureForecastingError(error_func, test_region, training_region)

        # will calculate the forecast error at each point of the region
        return measure_error.apply_to(forecast_region)


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
        self.test_len = solver_metadata.test_len
        self.error_type = solver_metadata.error_type

        # calculated during training
        self.partition = partition
        self.arima_model_region_training = arima_model_region_training
        self.arima_model_region_whole = arima_model_region_whole
        self.generalization_errors = generalization_errors

        # flag
        self.prepared = False

    def predict(self, prediction_region, forecast_len, output_home='outputs', plot=True):

        self.logger.debug('Predicting for region: {}'.format(prediction_region))

        if not self.prepared:
            self.prepare_for_predictions()

        # plot the whole region and the prediction region
        if plot:
            self.plot_regions(prediction_region, output_home)

        px1, px2, py1, py2 = prediction_region.x1, prediction_region.x2, prediction_region.y1, \
            prediction_region.y2

        # the generalization errors are precalculated, just subset them
        error_subregion = self.generalization_errors.region_subset(prediction_region)

        # in-sample (forecast_len=0) or out-of-sample (forecast_len > 0)?
        if forecast_len > 0:
            # this is meant to be an out-of-sample forecast
            is_out_of_sample = True

            # forecast the specified number of samples
            actual_forecast_len = forecast_len

            # this ARIMA model was trained with the whole dataset, so it will produce out-of-sample
            # predictions
            arima_model_region = self.arima_model_region_whole

            # no test data available for out-of-sample
            test_subregion = None

        else:
            # this is meant to be an in-sample forecast
            is_out_of_sample = False

            # the in-sample forecast is always the same length as the test samples
            actual_forecast_len = self.test_len

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
        forecast_subregion = arima_model_subset.apply_to(error_subregion, actual_forecast_len)

        # handle descaling here: we want to present descaled data to users
        if self.region_metadata.scaled:

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

        # create the result object using the builder:
        # is_out_of_sample True -> out-of-sample forecast
        # is_out_of_sample False -> in-sample forecast
        # the forecast_len will not be used by the builder for in-sample results
        builder = PredictionQueryResultBuilder(solver_metadata=self.metadata,
                                               forecast_len=actual_forecast_len,
                                               forecast_subregion=forecast_subregion,
                                               test_subregion=test_subregion,
                                               error_subregion=error_subregion,
                                               prediction_region=prediction_region,
                                               spt_region=self.spt_region,
                                               output_home=output_home,
                                               is_out_of_sample=is_out_of_sample)

        # add clustering information to results via the partition
        result = builder.with_partition(self.partition).build()
        return result

    def prepare_for_predictions(self):
        '''
        Prepare to attend prediction requests. For now, this is called on-demand.
        In the future, it can be used to lazily instantiate a solver.
        '''

        # load the region again
        # normally only to get shape, but for scaled regions it also has info useful for
        # descaling the data
        self.spt_region = self.region_metadata.create_instance()
        self.logger.debug('Loaded spt_region for predictions: {} {!r}'.format(self.spt_region,
                                                                              self.spt_region))

        # get test region using the specified number of past samples, apply scaling if needed
        splitter = SplitTrainingAndTestLast(self.test_len)
        (_, self.test_region) = splitter.split(self.spt_region)
        if self.test_region.has_scaling():
            self.test_region = self.test_region.descale()

        # set prepared flag
        self.prepared = True

    def plot_regions(self, prediction_region, output_home):
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
        fs_util.mkdir(self.metadata.output_dir(output_home))
        plt.draw()

        plot_name = self.get_plot_filename(prediction_region, output_home)
        plt.savefig(plot_name)
        self.logger.info('Saved figure: {}'.format(plot_name))

        # show figure
        plt.show()

    def get_plot_filename(self, prediction_region, output_home):
        '''
        Returns a string representing the filename for the plot.
        '''
        solver_plot_dir = self.metadata.output_dir(output_home)
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
            'Test samples: {}'.format(self.test_len),
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
                |- <clustering>
                    |- partition.pkl
                    |- auto_arima_<auto_arima_params>_model_region.pkl
                    |- auto_arima_<auto_arima_params>_errors_<error_type>.pkl
    '''

    def __init__(self, solver_metadata):

        # solver metadata
        self.metadata = solver_metadata

        self.region_metadata = self.metadata.region_metadata
        self.clustering_metadata = self.metadata.clustering_metadata
        self.distance_measure = self.metadata.distance_measure
        self.test_len = self.metadata.test_len
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
        partition = self.load_partition()

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
        self.logger.info('Loaded solver: {}'.format(self.metadata))

        return AutoARIMASolver(solver_metadata=self.metadata,
                               partition=partition,
                               arima_model_region_training=arima_model_region_training,
                               arima_model_region_whole=arima_model_region_whole,
                               generalization_errors=generalization_errors)

    def load_partition(self):
        '''
        Load a previously saved partititon, if available.
        Since the clustering algorithm handles its own persistence, this method will only try
        to load a partition without failing. If loading is not possible we calculate the partition
        again from the corresponding spatio-temporal region.

        This solves a corner case that can arise if a partition is deleted but we still want
        the models that have been calculated.
        '''
        # the clustering algorithm can manage partition persistence,
        # but need to create an instance of this algorithm to delegate task
        # TODO get a better handle for the functionality?
        clustering_factory = ClusteringFactory(self.distance_measure)
        clustering_algorithm = clustering_factory.instance(self.clustering_metadata)

        # if it is saved, use it
        partition = clustering_algorithm.try_load_previous_partition(self.region_metadata,
                                                                     pickle_home='pickle')

        if partition is None:
            # the partition was deleted T_T
            # calculate it again, for that we need the region
            spt_region = self.region_metadata.create_instance()
            partition = clustering_algorithm.partition(spt_region,
                                                       with_medoids=True,
                                                       pickle_home='pickle')
        return partition

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
        by the region, clustering, distance, auto arima metadata and number of test samples used.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        # arima_model_filename = 'model-region-training__{!r}.pkl'.format(
        # self.metadata.model_params)
        arima_model_filename = self.metadata.trained_model_region_filename('models-at-medoids')
        return os.path.join(solver_pickle_dir, arima_model_filename)

    def arima_model_whole_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region based on the whole dataset
        (as opposed to just the training dataset); given by the region, clustering, distance and
        auto arima metadata.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        # arima_model_filename = 'model-region-whole__{!r}.pkl'.format(self.metadata.model_params)
        arima_model_filename = self.metadata.trained_model_region_filename('models-at-medoids',
                                                                           with_test_samples=False)
        return os.path.join(solver_pickle_dir, arima_model_filename)

    def generalization_errors_pickle_path(self):
        '''
        Full path of the pickle object of the arima model region, given by the region, clustering,
        distance, auto arima metadata, test_len, also the error type.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()

        template = 'errors__{!r}__tp{}__{}.pkl'
        generalization_errors_filename = template.format(self.metadata.model_params,
                                                         self.test_len, self.error_type)
        return os.path.join(solver_pickle_dir, generalization_errors_filename)
