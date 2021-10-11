import numpy as np

from spta.clustering.kmedoids import KmedoidsClusteringMetadata
from spta.model.forecast import ForecastAnalysis
from spta.model.train import NoTrainer
from spta.region import TimeInterval
from spta.region.scaling import SpatioTemporalScaled
from spta.solver.model import SolverPickler
from spta.solver.result import PredictionQueryResultBuilder
from spta.solver.metadata import SolverMetadataBuilder

from spta.util import log as log_util

from .dnn import ClassifierDNN


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
    - model_trainer -> will be used to create a ModelRegion
    - model_params -> needed to find the trained models in the medoids
    - classifier_params -> needed to find and configure the external classifier
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
                 model_params, classifier_params, test_len, error_type):
        self.region_metadata = region_metadata
        self.distance_measure = distance_measure
        self.clustering_suite = clustering_suite
        self.model_trainer = model_trainer
        self.model_params = model_params
        self.classifier_params = classifier_params
        self.test_len = test_len
        self.error_type = error_type

        self.classifier = None

        # cache solvers so that we don't reconstruct them every single time
        self.solvers = {}

    def prepare(self, out_of_sample_region):
        '''
        Here we make sure that the classifier is loaded so that it can answer with a label.
        The target_spt_region is used to extract the series used on the classifier to retrieve a label.
        '''
        # need actual code...
        # self.classifier = RandomClassifier()

        self.classifier = ClassifierDNN(self.classifier_params)

        self.spt_region = self.region_metadata.create_instance()

        if out_of_sample_region is None:
            # no out-of-sample target specified, us the same region
            self.target_spt_region = self.spt_region
        else:
            # use a different, 'out-of-sample' region
            self.target_spt_region = out_of_sample_region.create_instance()

        metadata_builder = SolverMetadataBuilder(region_metadata=self.region_metadata,
                                                 model_params=self.model_params,
                                                 test_len=self.test_len,
                                                 error_type=self.error_type)
        self.solver_metadata = metadata_builder.for_suite(self.clustering_suite, self.distance_measure).build()

    def predict(self, prediction_region, tp, out_of_sample_region=None, output_home='outputs'):
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
            self.prepare(out_of_sample_region)
        assert self.classifier is not None
        assert self.target_spt_region is not None

        # get d(point, medoid labels)
        classifier_labels_by_point = self.get_labels_from_classifier(prediction_region, tp)

        # create d(point, medoid models)
        models_by_point = {}
        for point, classifier_label in classifier_labels_by_point.items():
            solver_metadata = self.solver_metadata_for_classifier_label(classifier_label)
            models_by_point[point] = self.retrieve_model_for_solver(solver_metadata, classifier_label)
            self.logger.debug('Adding model {} at {}'.format(classifier_label, point))

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
        series_len, _, _ = self.target_spt_region.shape
        ti = TimeInterval(series_len - tp, series_len)
        region_of_interest = self.target_spt_region.subset(prediction_region, ti)

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
            if hasattr(solver, 'partition'):
                self.logger.debug('Partition: {} {}'.format(solver.partition, solver.partition.medoids))

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
        # note that the points are relative to the prediction region, so they are already zero-based
        for point, model in models_by_point.items():
            numpy_model_array[point.x][point.y] = model

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


if __name__ == '__main__':
    from spta.arima import AutoArimaParams
    from spta.arima.train import TrainerAutoArima
    from spta.dataset.metadata import TemporalMetadata, SamplesPerDay
    from spta.distance.dtw import DistanceByDTW
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.clustering.kmedoids import kmedoids_metadata_generator
    from spta.classifier.dnn import ClassifierDNNParams

    logger = log_util.setup_log('DEBUG')

    dataset_class_name = 'spta.dataset.csfr.DatasetCSFR'
    temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
    region_metadata = SpatioTemporalRegionMetadata(name='nordeste_small',
                                                   region=Region(40, 50, 50, 60),
                                                   temporal_md=temporal_md,
                                                   dataset_class_name=dataset_class_name,
                                                   scaled=False)

    classifier_params = ClassifierDNNParams('lstm_wSW_label_k-seed-ci', 'lstm/model_lstm_wSW_label_k-seed-ci.h5', 'lstm/labels_k-seed-ci.csv', 15)
    model_params = AutoArimaParams(1, 1, 3, 3, None, True)
    test_len = 8
    error_type = 'sMAPE'
    distance_measure = DistanceByDTW()

    clustering_suite = kmedoids_metadata_generator(k_values=range(2, 4), seed_values=range(0, 2))
    clustering_suite.identifier = 'quick'

    for item in clustering_suite:
        logger.debug(item)

    model_trainer = TrainerAutoArima(model_params, region_metadata.x_len, region_metadata.y_len)
    solver = SolverFromClassifier(region_metadata=region_metadata,
                                  distance_measure=distance_measure,
                                  clustering_suite=clustering_suite,
                                  model_trainer=model_trainer,
                                  model_params=model_params,
                                  classifier_params=classifier_params,
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
