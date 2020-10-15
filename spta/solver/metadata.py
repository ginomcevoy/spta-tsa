import os


class SolverMetadataBasic(object):
    '''
    The metadata of a basic solver that only has the region, model parameters and error type.
    '''

    def __init__(self, region_metadata, model_params, test_len, error_type):
        self.region_metadata = region_metadata
        self.model_params = model_params
        self.test_len = test_len
        self.error_type = error_type

    def output_dir(self, output_home):
        '''
        Directory to store outputs relevant to this solver metadata.
        '''
        output_dir_region = self.region_metadata.output_dir(output_home)
        model_params_dir = '{!r}'.format(self.model_params)
        return os.path.join(output_dir_region, model_params_dir)

    def csv_dir(self, output_home, prediction_region):
        '''
        Returns a string representing the directory for storing CSV files of this solver.
        Uses the prediction region as part of the dir.
        '''
        solver_output_dir = self.output_dir(output_home)
        csv_subdir = self.prediction_region_as_str(prediction_region)
        return os.path.join(solver_output_dir, csv_subdir)

    def csv_filename(self, name_prefix, prediction_region, forecast_len):
        '''
        Returns a string representing a CSV file for this query.
        The prefix can be used to differentiate between the CSV of the query and the CSV of the
        query summary.
        '''
        prediction_region_str = self.prediction_region_as_str(prediction_region)
        return '{}__{}__tp{}__tf{}__{}.csv'.format(name_prefix, prediction_region_str,
                                                   self.test_len, forecast_len, self.error_type)

    def pickle_dir(self, pickle_home='pickle'):
        return self.region_metadata.pickle_dir(pickle_home)

    def trained_model_region_filename(self, name_prefix, with_test_samples=True):
        '''
        Calculates a string representing a filename to store a trained model region as pickle.

        A trained model region can have models trained at each point of the region, or only
        at some representatives (e.g. medoids). The name prefix can be used to differentiate the
        different cases.

        By default, it is assumed that the models were trained by reserving some test samples
        given by test_len, this is reflected in the filename. If a model region is later retrieved,
        for forecasting, an adequate test_len has to be specified by the user.

        If a model region was trained with the full dataset, then use with_test_samples=False,
        the filename will not include test_len.
        '''
        test_len_str = 'tp{}'.format(self.test_len)
        if not with_test_samples:
            # assume that the models are using the whole dataset (no test subset)
            test_len_str = 'whole'

        return '{}__{!r}__{}.pkl'.format(name_prefix, self.model_params, test_len_str)

    def prediction_region_as_str(self, prediction_region):
        '''
        Used internally for csv_dir and csv_filename.
        '''
        return 'region-{}-{}-{}-{}'.format(prediction_region.x1,
                                           prediction_region.x2,
                                           prediction_region.y1,
                                           prediction_region.y2)

    def __str__(self):
        return '{} {} tp{} {}'.format(self.region_metadata, self.model_params, self.test_len,
                                      self.error_type)


class MetadataWithClustering(SolverMetadataBasic):
    '''
    Reifies decorator pattern to add support for solvers that use clustering.
    '''

    def __init__(self, solver_metadata, clustering_metadata, distance_measure):
        self.decorated = solver_metadata

        # basic solver data
        self.region_metadata = self.decorated.region_metadata
        self.model_params = self.decorated.model_params
        self.error_type = self.decorated.error_type
        self.test_len = self.decorated.test_len

        # clustering-specific
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure

    def output_dir(self, output_home):
        '''
        Directory to store outputs relevant to this solver metadata.
        This implementation uses region metadata, clustering metadata and the distance measure.
        TODO: should distance measure be part of the clustering metadata?
        '''
        output_dir_clustering = self.clustering_metadata.output_dir(output_home,
                                                                    self.region_metadata,
                                                                    self.distance_measure)
        model_params_dir = '{!r}'.format(self.model_params)
        return os.path.join(output_dir_clustering, model_params_dir)

    def csv_filename(self, name_prefix, prediction_region, forecast_len):
        '''
        Returns a string representing a CSV file for this query.
        The prefix can be used to differentiate between the CSV of the query and the CSV of the
        query summary.

        If forecast_len is 0, it is assumed that the query is for in-sample forecast using the same
        number of samples as test_len, the name reflects this (tfis)
        '''
        if forecast_len > 0:
            # out-of-sample forecast
            forecast_len_str = '{}'.format(forecast_len)
        else:
            # in-sample forecast
            forecast_len_str = '_is'

        prediction_region_str = self.prediction_region_as_str(prediction_region)
        return '{}-{!r}__{}__tp{}__tf{}__{}.csv'.format(name_prefix, self.clustering_metadata,
                                                        prediction_region_str, self.test_len,
                                                        forecast_len_str, self.error_type)

    def pickle_dir(self):
        return self.clustering_metadata.pickle_dir(self.region_metadata, self.distance_measure)

    def __str__(self):
        return '{} {} {} {} {}'.format(self.region_metadata, self.clustering_metadata,
                                       self.distance_measure, self.model_params, self.error_type)

    def __repr__(self):
        return '{!r}__{!r}__{!r}__{!r}__{}'.format(self.region_metadata, self.clustering_metadata,
                                                   self.distance_measure, self.model_params,
                                                   self.error_type)


class SolverMetadataBuilder():
    '''
    Reifies builder pattern to create metadata instances.
    '''

    def __init__(self, region_metadata, model_params, test_len, error_type):
        self.metadata = SolverMetadataBasic(region_metadata, model_params, test_len, error_type)

    def with_clustering(self, clustering_metadata, distance_measure):
        self.metadata = MetadataWithClustering(self.metadata, clustering_metadata,
                                               distance_measure)
        return self

    def build(self):
        return self.metadata
