import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from spta.clustering.factory import ClusteringFactory
from spta.clustering.kmedoids import KmedoidsClusteringMetadata

from spta.model.forecast import ForecastAnalysis
from spta.model.train import NoTrainer, SplitTrainingAndTestLast

from spta.region import TimeInterval
from spta.region.scaling import SpatioTemporalScaled

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

from .metadata import SolverMetadataBuilder
from .result import PredictionQueryResultBuilder


class SolverWithMedoids(log_util.LoggerMixin):
    '''
    Represents a cluster partition of a given spatio-temporal region, with models trained only
    at the medoids. It can answer a forecast query of a specified subregion, by using, at each
    point in the subregion, the forecast at the medoid of the cluster for which each point is
    a member.
    '''

    def __init__(self, solver_metadata, partition, model_region_training,
                 model_region_whole, generalization_errors):

        # user input
        self.metadata = solver_metadata
        self.region_metadata = solver_metadata.region_metadata
        self.clustering_metadata = solver_metadata.clustering_metadata
        self.distance_measure = solver_metadata.distance_measure
        self.model_params = solver_metadata.model_params
        self.test_len = solver_metadata.test_len
        self.error_type = solver_metadata.error_type

        # calculated during training
        self.partition = partition
        self.model_region_training = model_region_training
        self.model_region_whole = model_region_whole
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

            # this model was trained with the whole dataset, so it will produce out-of-sample predictions
            model_region_to_use = self.model_region_whole

            # no test data available for out-of-sample
            test_subregion = None

        else:
            # this is meant to be an in-sample forecast
            is_out_of_sample = False

            # the in-sample forecast is always the same length as the test samples
            actual_forecast_len = self.test_len

            # this model was trained with the training dataset, so it will produce in-sample
            # predictions (comparable to the test)
            model_region_to_use = self.model_region_training

            # extract the test data for the prediction region
            test_subregion = self.test_region.region_subset(prediction_region)

        # compute the prediction: create a forecast over the region
        # to do this, obtain a subset of the model region
        model_numpy_subregion = model_region_to_use.as_numpy[px1:px2, py1:py2]
        model_subregion = model_region_to_use.instance(model_numpy_subregion)

        # here we just need an empty region with the correct shape
        # and the error_subregion is right there... so use it
        # TODO will this approach break with other models?
        forecast_subregion = model_subregion.apply_to(error_subregion, actual_forecast_len)

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
        See SolverPickler for details.
        '''
        # delegate to the pickler
        pickler = SolverPickler(solver_metadata=self.metadata)
        pickler.save_solver(self)

    def __str__(self):
        '''
        Solver string representation in multiple lines.
        '''
        lines = [
            'Region: {}'.format(self.region_metadata),
            'Model: {}'.format(self.model_params),
            '{}'.format(self.clustering_metadata),
            'Test samples: {}'.format(self.test_len),
            'Error function: {}'.format(self.error_type)
        ]
        return '\n'.join(lines)


class SolverFromClassifier(log_util.LoggerMixin):
    '''
    A solver that calculates the result of a prediction query using the outcome of an external
    classifier (e.g. Neural Network). The classifier is a function that, given an input series of size
    tp, a clustering suite and a region, returns, for each point, one of the medoids from any of
    the clustering metadata that are available in the clustering suite.

    classifier(s, cs, r) = [(md_i1, m_ij1), (md_i2, m_ij2), ..., (md_iN, m_ijN)]
    classifier = f: (R^p, CS, D) -> (CS, M)^N, where:

    - s in R^p is a time series of size tp
    - cs is one of the available clustering suites of the CS space
    - r is a 2d rectangle in D specified by its 4 coordinates such that it has N points
    - md_il in clustering_suite CS is a metadata for a clustering, e.g (k=10, seed=0, mode=lite)
      for k-medoids
    - m_ijl in M is one of the medoids in the partitioning specified by the clustering metadata md_il.

    Here, the following assumptions are made:

    - Assume that the classifier has given us the (md_iP, m_ijP) tuple for the series s in point P.
      The task of the solver is then to retrieve the model of the medoid m_ij, that was saved when
      training a solver using md_i and the model parameters.

    - Assume that the appropriate solver has been previously trained and saved as a pickle object:
      if the model is not found, an error is returned. Also the generalization error needs to be available.

    - We assume a constant size tf for the predicted series. The output is obtained calling the
      corresponding model to make a prediction of size tf.

    The inputs are as follows:
    - region metadata
    - distance_measure
    - model_trainer
    - model_params
    - a region R for the prediction query
    - for each point P in R, the output of the classifier, a list of (md_i, m_j) tuples.
    - the desired prediction size tf.

    The output is, for each point P in R:
    - the predicted series of size tf
    - the generalization error of the model

    TODO if using sliding window, we also need to use an offset and model.predict() instead of
    model.forecast(), in order to make an in-sample prediction.
    '''

    def __init__(self, region_metadata, distance_measure, clustering_suite, model_trainer,
                 model_params, test_len, error_type):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.clustering_suite = clustering_suite
        self.model_trainer = model_trainer
        self.model_params = model_params
        self.test_len = test_len
        self.error_type = error_type

        self.classifier = None

        # cache solvers so that we don't reconstruct them every single time
        self.solvers = {}

    def prepare(self):
        '''
        Here we make sure that the classifier is loaded so that it can answer with a label.
        '''
        # need actual code...
        self.classifier = RandomClassifier()
        self.spt_region = self.region_metadata.create_instance()

        metadata_builder = SolverMetadataBuilder(region_metadata=self.region_metadata,
                                                 model_params=self.model_params,
                                                 test_len=self.test_len,
                                                 error_type=self.error_type)
        self.solver_metadata = metadata_builder.for_suite(self.clustering_suite, self.distance_measure).build()

    def predict(self, prediction_region, tp, output_home='outputs'):
        '''
        Ask the solver to create a forecast for the specified region using the specified number
        of past points as reference.

        This will:

        1. call the classifier interface to get the medoid labels for each point
        2. use SolverPickler to retrieve the medoid models from a trained solver
        3. create a ModelRegion for the prediction region with the medoid models
        4. Follow a similar implementation of experiments.auto_arima.solver_each to create a
           PredictionQueryResult instance.
        '''

        if self.classifier is None:
            self.prepare()
        assert self.classifier is not None
        assert self.spt_region is not None

        # get d(point, medoid labels)
        classifier_labels_by_point = self.get_labels_from_classifier(prediction_region, tp)

        # create d(point, medoid models)
        models_by_point = {}
        for point, classifier_label in classifier_labels_by_point.items():
            solver_metadata = self.solver_metadata_for_classifier_label(classifier_label)
            models_by_point[point] = self.retrieve_model_for_solver(solver_metadata, classifier_label)

        # ModelRegion instance
        model_region = self.build_model_region_for_models(prediction_region, models_by_point)

        prediction_result = self.create_prediction_query_result(prediction_region=prediction_region,
                                                                model_region=model_region,
                                                                classifier_labels_by_point=classifier_labels_by_point,
                                                                output_home=output_home)
        return prediction_result

    def get_labels_from_classifier(self, prediction_region, tp):

        # extract subseries of length tp from our data, this will be the input for the classifier
        # data is from the last tp points
        series_len, _, _ = self.spt_region.shape
        ti = TimeInterval(series_len - tp, series_len)
        region_of_interest = self.spt_region.subset(prediction_region, ti)

        classifier_labels_by_point = {}

        # the classifier works for each point, so we can use the spatio-temporal iterator here
        for point, series_of_len_tp in region_of_interest:
            classifier_labels_by_point[point] = self.classifier.label_for_series(series_of_len_tp)

        return classifier_labels_by_point

    def solver_metadata_for_classifier_label(self, classifier_label):
        '''
        Given a label that is the output of the classifier, create a SolverMetadata istance
        that can later be used to retrieve the model for one of its medoids.

        Assuming kmedoids!
        Assuming mode=lite!
        '''
        clustering_metadata = KmedoidsClusteringMetadata.from_classifier_label(classifier_label)

        builder = SolverMetadataBuilder(region_metadata=self.region_metadata,
                                        model_params=self.model_params,
                                        test_len=self.test_len,
                                        error_type=self.error_type)
        builder.with_clustering(clustering_metadata, self.distance_measure)
        solver_metadata = builder.build()
        return solver_metadata

    def retrieve_model_for_solver(self, solver_metadata, classifier_label):
        '''
        Use an instance of SolverMetadataWithClustering to load a previously saved solver,
        and retrieve the saved model at the medoid representing the cluster index
        that is encoded in the classifier_label.

        Returns the 'training model', which has been trained with (series_len - test_len) points.
        This model is meant to do 'in-sample forecast'.
        '''

        solver_repr = repr(solver_metadata)
        self.logger.debug('Looking for solver metadata: {}'.format(solver_repr))
        if solver_repr in self.solvers:
            solver = self.solvers[solver_repr]
            self.logger.debug('Using recovered solver:\n{}'.format(solver))
        else:
            # new solver
            solver_pickler = SolverPickler(solver_metadata)
            solver = solver_pickler.load_solver()
            self.solvers[solver_repr] = solver
            self.logger.debug('Adding new solver:\n{}'.format(solver))

        # get the cluster_index from the label:
        # 13-0-6 -> 6
        cluster_index_str = classifier_label.split('-')[2]
        cluster_index = int(cluster_index_str)

        # find the medoid given cluster_index, return its model
        chosen_medoid = solver.partition.medoids[cluster_index]
        return solver.model_region_training.value_at(chosen_medoid)

    def build_model_region_for_models(self, prediction_region, models_by_point):
        '''
        Return an instance of ModelRegion that matches the prediction region.
        For each point, it will contain the model of the medoid corresponding to the
        classifier label for that point.
        '''
        # the ModelRegion needs to be created 'manually', by collecting each of the medoids
        x_len = prediction_region.x2 - prediction_region.x1
        y_len = prediction_region.y2 - prediction_region.y1
        numpy_model_array = np.empty((x_len, y_len), dtype=object)
        self.logger.debug('numpy_model_array shape: {}'.format(numpy_model_array.shape))

        # iterate each {point, model} key-value pair
        # note that the point is relative to spt_region, but the numpy array is zero-based
        for point, model in models_by_point.items():
            # get zero-based coords
            x_numpy = point.x - prediction_region.x1
            y_numpy = point.y - prediction_region.y1
            numpy_model_array[x_numpy][y_numpy] = model

        # the trainer knows how to create the correct region for the array of models
        return self.model_trainer.create_model_region(numpy_model_array)

    def create_prediction_query_result(self, prediction_region, model_region,
                                       classifier_labels_by_point, output_home):

        # subset to get prediction spatio-temporal region
        prediction_spt_region = self.spt_region.region_subset(prediction_region)

        # delegate forecasting to this implementation, does not do actual training and takes care
        # of forecast/error details
        no_trainer = NoTrainer(model_region)
        forecast_analysis = ForecastAnalysis(no_trainer, parallel_workers=None)
        forecast_analysis.train_models(prediction_spt_region, self.test_len)

        # do in-sample forecasting with models at each point, evaluate error
        forecast_region_each, error_region_each, time_forecast = \
            forecast_analysis.forecast_at_each_point(forecast_len=self.test_len, error_type=self.error_type)

        # also need the test subregion for results
        test_subregion = forecast_analysis.test_region

        # handle descaling here: we want to present descaled data to users
        if self.region_metadata.scaled:

            forecast_region_each = self.descale_subregion(forecast_region_each, prediction_spt_region)
            test_subregion = self.descale_subregion(test_subregion, prediction_spt_region)

        # prepare the results with the data gathered
        # is_future is False, we always use in-sample forecasting here
        result_builder = PredictionQueryResultBuilder(solver_metadata=self.solver_metadata,
                                                      forecast_len=self.test_len,
                                                      forecast_subregion=forecast_region_each,
                                                      test_subregion=test_subregion,
                                                      error_subregion=error_region_each,
                                                      prediction_region=prediction_region,
                                                      spt_region=self.spt_region,
                                                      output_home='outputs',
                                                      is_out_of_sample=False)
        prediction_result = result_builder.from_classifier(self.clustering_suite, classifier_labels_by_point).build()
        return prediction_result

    def descale_subregion(self, subregion, prediction_spt_region):
        '''
        The forecast_region_each and test_subregion are not aware of the scaling.
        As a workaround, use the spatio-temporal region of the prediction subset
        (which HAS the scaling data) and to retrieve appropriate descaling info.
        '''
        self.logger.debug('About to descale manually: {}'.format(subregion))
        subregion_with_scaling = SpatioTemporalScaled(subregion,
                                                      scale_min=prediction_spt_region.scale_min,
                                                      scale_max=prediction_spt_region.scale_max)
        return subregion_with_scaling.descale()


class MockClassifier(object):
    '''
    A mock classifier that always returns the same label, for a clustering partition that was used
    to train a solver.
    '''

    def label_for_series(self, series_of_len_tp):
        return '2-0-1'


class RandomClassifier(object):
    '''
    A mock classifier that returns one of two labels, for two clustering partitions that were used to
    train solvers.
    '''

    def label_for_series(self, series_of_len_tp):
        randn = np.random.choice(2)
        medoid_index = int(np.floor(randn))
        return '2-0-{}'.format(medoid_index)


class SolverPickler(log_util.LoggerMixin):
    '''
    Handles the persistence of a solver using pickle.
    The metadata is used to create a directory structure that uniquely identies the metadata
    parameters: different parameters should produce a different pickle file.

    Example:

    |- pickle
        |- <region>
            |- dtw
                |- <clustering>
                    |- errors__<model_params>__tp<tp>__<error>.pkl
                    |- partition_<clustering_metadata>.pkl
                    |- models-at-medoids-<model_params>__tp<tp>.pkl
                    |- models-at-medoids-<model_params>__whole.pkl
    '''

    def __init__(self, solver_metadata):

        # solver metadata
        self.metadata = solver_metadata

        self.region_metadata = self.metadata.region_metadata
        self.clustering_metadata = self.metadata.clustering_metadata
        self.distance_measure = self.metadata.distance_measure
        self.test_len = self.metadata.test_len
        self.error_type = self.metadata.error_type

    def save_solver(self, solver):
        '''
        persist the solver details as pickle objects.
        '''
        model_region_training = solver.model_region_training
        model_region_whole = solver.model_region_whole
        generalization_errors = solver.generalization_errors

        # create the directory
        solver_pickle_dir = self.metadata.pickle_dir()
        fs_util.mkdir(solver_pickle_dir)

        # the clustering algorithm is already managing partition persistence when it creates it

        # save the  model region based on training dataset
        model_training_path = self.model_training_pickle_path()
        with open(model_training_path, 'wb') as pickle_file:
            pickle.dump(model_region_training, pickle_file)
            self.logger.debug('Saved model (training): {}'.format(model_training_path))

        # save the model region for whole dataset
        model_whole_path = self.model_whole_pickle_path()
        with open(model_whole_path, 'wb') as pickle_file:
            pickle.dump(model_region_whole, pickle_file)
            self.logger.debug('Saved model (whole): {}'.format(model_whole_path))

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

        # load the model region based on training data
        model_training_path = self.model_training_pickle_path()
        self.logger.debug('Loading models (training): {}'.format(model_training_path))
        with open(model_training_path, 'rb') as pickle_file:
            model_region_training = pickle.load(pickle_file)

        # load the model region based on whole data
        model_whole_path = self.model_whole_pickle_path()
        self.logger.debug('Loading models (whole): {}'.format(model_whole_path))
        with open(model_whole_path, 'rb') as pickle_file:
            model_region_whole = pickle.load(pickle_file)

        # load the generalization errors
        generalization_errors_path = self.generalization_errors_pickle_path()
        self.logger.debug('Attempting to errors at {}'.format(generalization_errors_path))
        with open(generalization_errors_path, 'rb') as pickle_file:
            generalization_errors = pickle.load(pickle_file)

        # recreate the solver
        self.logger.info('Loaded solver: {}'.format(self.metadata))

        return SolverWithMedoids(solver_metadata=self.metadata,
                                 partition=partition,
                                 model_region_training=model_region_training,
                                 model_region_whole=model_region_whole,
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

    def model_training_pickle_path(self):
        '''
        Full path of the pickle object of the model region based on training dataset; given
        by the region, clustering, distance, model parameters and number of test samples used.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        model_filename = self.metadata.trained_model_region_filename('models-at-medoids')
        return os.path.join(solver_pickle_dir, model_filename)

    def model_whole_pickle_path(self):
        '''
        Full path of the pickle object of the model region based on the whole dataset
        (as opposed to just the training dataset); given by the region, clustering, distance and
        model parameters.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()
        model_filename = self.metadata.trained_model_region_filename('models-at-medoids',
                                                                     with_test_samples=False)
        return os.path.join(solver_pickle_dir, model_filename)

    def generalization_errors_pickle_path(self):
        '''
        Full path of the pickle object of the model region, given by the region, clustering,
        distance, model parameters, test_len, also the error type.
        '''
        solver_pickle_dir = self.metadata.pickle_dir()

        template = 'errors__{!r}__tp{}__{}.pkl'
        generalization_errors_filename = template.format(self.metadata.model_params,
                                                         self.test_len, self.error_type)
        return os.path.join(solver_pickle_dir, generalization_errors_filename)


if __name__ == '__main__':
    from spta.arima import AutoArimaParams
    from spta.arima.train import TrainerAutoArima
    from spta.distance.dtw import DistanceByDTW
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.clustering.kmedoids import kmedoids_metadata_generator

    log_util.setup_log('DEBUG')

    region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(40, 50, 50, 60),
                                                   2015, 2015, 1, scaled=False)
    model_params = AutoArimaParams(1, 1, 3, 3, None, True)
    test_len = 8
    error_type = 'sMAPE'
    distance_measure = DistanceByDTW()

    clustering_suite = kmedoids_metadata_generator(k_values=range(2, 4), seed_values=range(0, 2))
    clustering_suite.identifier = 'quick'

    model_trainer = TrainerAutoArima(model_params, region_metadata.x_len, region_metadata.y_len)
    solver = SolverFromClassifier(region_metadata=region_metadata,
                                  distance_measure=distance_measure,
                                  clustering_suite=clustering_suite,
                                  model_trainer=model_trainer,
                                  model_params=model_params,
                                  test_len=test_len,
                                  error_type=error_type)

    prediction_region = Region(1, 3, 1, 4)
    tp = 15
    prediction_result = solver.predict(prediction_region, tp, output_home='outputs')

    # for printing forecast and error values
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    for relative_point in prediction_result:

        # for each point, the result can print a text output
        print('*********************************')
        text = prediction_result.lines_for_point(relative_point)
        print('\n'.join(text))

    # the result can save relevant information to CSV
    prediction_result.save_as_csv()
