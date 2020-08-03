import os


class SolverMetadataBasic(object):
    '''
    The metadata of a basic solver that only has the region, model parameters and error type.
    '''

    def __init__(self, region_metadata, model_params, error_type):
        self.region_metadata = region_metadata
        self.model_params = model_params
        self.error_type = error_type

    def output_dir(self, output_home):
        '''
        Directory to store outputs relevant to this solver metadata.
        '''
        output_dir_clustering = self.region_metadata.output_dir(output_home)
        model_params_dir = '{!r}'.format(self.model_params)
        return os.path.join(output_dir_clustering, model_params_dir)

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
        return '{}__{}__f{}__{}.csv'.format(name_prefix,
                                            self.prediction_region_as_str(prediction_region),
                                            forecast_len, self.error_type)

    def pickle_dir(self, pickle_home='pickle'):
        return self.region_metadata.pickle_dir(pickle_home)

    def prediction_region_as_str(self, prediction_region):
        '''
        Used internally for csv_dir and csv_filename.
        '''
        return 'region-{}-{}-{}-{}'.format(prediction_region.x1,
                                           prediction_region.x2,
                                           prediction_region.y1,
                                           prediction_region.y2)

    def __str__(self):
        return '{} {} {}'.format(self.region_metadata, self.model_params, self.error_type)


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

        # clustering-specific
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure

    def output_dir(self, output_home):
        '''
        Directory to store outputs relevant to this solver metadata.
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
        '''
        return '{}-{!r}__{}__f{}__{}.csv'.format(name_prefix, self.clustering_metadata,
                                                 self.prediction_region_as_str(prediction_region),
                                                 forecast_len, self.error_type)

    def pickle_dir(self):
        return self.clustering_metadata.pickle_dir(self.region_metadata, self.distance_measure)

    def __str__(self):
        return '{} {} {} {} {}'.format(self.region_metadata, self.clustering_metadata,
                                       self.distance_measure, self.model_params, self.error_type)


class SolverMetadataBuilder():
    '''
    Reifies builder pattern to create metadata instances.
    '''

    def __init__(self, region_metadata, model_params, error_type):
        self.metadata = SolverMetadataBasic(region_metadata, model_params, error_type)

    def with_clustering(self, clustering_metadata, distance_measure):
        self.metadata = MetadataWithClustering(self.metadata, clustering_metadata,
                                               distance_measure)
        return self

    def build(self):
        return self.metadata
