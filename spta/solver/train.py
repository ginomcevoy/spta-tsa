from spta.clustering.factory import ClusteringFactory
from spta.model.error import MeasureForecastingError, get_error_func
from spta.model.train import SplitTrainingAndTestLast, TrainAtRepresentatives

from spta.util import log as log_util

from .metadata import SolverMetadataBuilder
from .model import SolverWithMedoids


class SolverTrainer(log_util.LoggerMixin):
    '''
    Prepares a SolverWithMedoids instance by partitioning the region and training models at the medoids
    of the resulting clusters.

    The solver expects two model regions: model_region_training and model_region_whole. To create
    model_region_training, a ModelTrainer instance is used. To create model_region_whole, the obtained
    models are refitted using a model_refitter instace.

    This implementation assumes that the trainer has a constant value for model_params valid for the
    entire region. For cases like auto ARIMA, where the model parameters change from point to point,
    the auto arima parameters (grid search parameters) are constant.
    '''

    def __init__(self, region_metadata, clustering_metadata, distance_measure, model_trainer,
                 model_params, test_len, error_type):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.model_trainer = model_trainer
        self.model_params = model_params
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
                                        model_params=self.model_params,
                                        test_len=self.test_len,
                                        error_type=self.error_type)
        self.metadata = builder.with_clustering(clustering_metadata=self.clustering_metadata,
                                                distance_measure=self.distance_measure).build()

        self.prepared = True

    def train(self, output_home='outputs'):
        '''
        Partition the region into clusters, then train the models on the cluster medoids.

        The models are trained using the training samples, and the test samples (out-of-time for
        the trained models) are used to calculated the forecast error. After this is done, the
        models are re-fitted using the full dataset, in order to improve out-of-sample forecast,
        called the 'future' scenario.

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
        #     This step is achieved by train_models_at_medoids().
        #
        #  2. Use the partition to create k clusters, each containing the trained model at its
        #     medoid, then merging the clusters with the values at the representatives.
        #     This step is achieved by replicate_representative_models().
        #

        # do 1. here
        medoid_model_region_training = self.train_models_at_medoids(training_region, medoids)

        # do 2. here
        replicated_model_region_training = \
            self.replicate_representative_models(medoid_model_region_training, partition, medoids)

        # calculate the generalization error for these ARIMA models: it is the prediction error
        # that is calculated using the test dataset.
        generalization_errors = self.calculate_errors(replicated_model_region_training,
                                                      training_region, test_region)
        self.logger.info('Overall error: {:.4f}'.format(generalization_errors.overall_error))

        # If the user sets the future flag for prediction, we want to use the entire series
        # to make out-of-sample predictions (out of sample)
        # For that, we need to re-train the models at the medoids again, these time using the
        # full dataset (not the training subset, which uses a series subset)
        # Note that we must use medoid_model_region_training instead of replicated_model_region_training,
        # otherwise we would be re-training x_len * y_len models instead of just k!
        # arima_refit_function = TrainerRefitArima(medoid_model_region_training)
        # arima_medoid_models_whole = arima_refit_function.apply_to(spt_region)
        model_refitter = self.model_trainer.create_refitter(medoid_model_region_training)
        medoid_model_region_whole = model_refitter.apply_to(spt_region)

        # replicate these refitted models, similar to step 2. above
        replicated_model_region_whole = \
            self.replicate_representative_models(medoid_model_region_whole, partition, medoids)

        # create a solver with the data acquired, this solver can answer queries
        solver = SolverWithMedoids(solver_metadata=self.metadata,
                                   partition=partition,
                                   model_region_training=replicated_model_region_training,
                                   model_region_whole=replicated_model_region_whole,
                                   generalization_errors=generalization_errors)
        return solver

    def train_models_at_medoids(self, training_region, medoids):
        '''
        Train the model region that will be used for forecasting has only k models.
        The model from each medoid will be replicated throughout its cluster later.
        '''
        # auto_arima_trainer = TrainerAutoArima(self.model_params, x_len, y_len)
        # auto_arima_trainer_at_medoids = TrainAtRepresentatives(auto_arima_trainer, medoids)
        # return auto_arima_trainer_at_medoids.apply_to(training_region)

        # decorate the trainer so that only the medoids are trained
        trainer_at_medoids = TrainAtRepresentatives(self.model_trainer, medoids)
        trained_models = trainer_at_medoids.apply_to(training_region)
        return trained_models

    def replicate_representative_models(self, medoid_model_region, partition, medoids):
        '''
        Given a sparse model region, replicate the models at each medoid over all the points
        of their respective clusters.
        '''
        # the clusters determine how to replicate the k models
        spatial_clusters = partition.create_all_spatial_clusters(medoid_model_region)

        # Merge the clusters into a single region by replicating the models at the medoids.
        # Note that this is just a spatial region
        merged_spatial_region = partition.merge_with_representatives_2d(spatial_clusters, medoids)

        # We want an instance of ModelRegion that can be applied to produce forecasts
        # the medoid_model_region is a ModelRegion of the correct subclass, it can create a new instance
        return medoid_model_region.instance(merged_spatial_region.as_numpy)

    def calculate_errors(self, model_region_training, training_region, test_region):
        '''
        Calculate the generalization error for the medoid models: it is the prediction error
        that is calculated using the test dataset. For these models, the test data is
        out-of-sample, because they were not trained with it.

        The number of forecast samples is fixed to be equal to the length of the test series,
        determined by the user request.
        '''
        # forecast the same number of samples for which we have test data
        forecast_len = test_region.series_len

        # create a forecast: for this we need a spatial_region of the right type,
        # use the test region for convenience
        forecast_region = model_region_training.apply_to(test_region, forecast_len)

        # calculate the error using a generic error function, requires the error type
        error_func = get_error_func(self.error_type)
        measure_error = MeasureForecastingError(error_func, test_region, training_region)

        # will calculate the forecast error at each point of the region
        return measure_error.apply_to(forecast_region)
