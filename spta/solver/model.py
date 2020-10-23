import matplotlib.pyplot as plt
import os
import pickle

from spta.clustering.factory import ClusteringFactory
from spta.model.train import SplitTrainingAndTestLast
from spta.region.scaling import SpatioTemporalScaled

from spta.util import fs as fs_util
from spta.util import log as log_util
from spta.util import plot as plot_util

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
