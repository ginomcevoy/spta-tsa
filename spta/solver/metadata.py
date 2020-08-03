import os


class SolverMetadata(object):

    def __init__(self, region_metadata, clustering_metadata, distance_measure, model_params,
                 error_type):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.model_params = model_params
        self.error_type = error_type

    def output_dir(self, output_prefix):
        output_dir_clustering = self.clustering_metadata.output_dir(output_prefix,
                                                                    self.region_metadata,
                                                                    self.distance_measure)
        model_params_dir = '{!r}'.format(self.model_params)
        return os.path.join(output_dir_clustering, model_params_dir)

    def csv_dir(self, output_prefix, prediction_region):
        '''
        Returns a string representing the directory for storing CSV files of this solver.
        Uses the prediction region as part of the dir.
        '''
        solver_output_dir = self.output_dir(output_prefix)
        csv_subdir = self.prediction_region_as_str(prediction_region)
        return os.path.join(solver_output_dir, csv_subdir)

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

    def prediction_region_as_str(self, prediction_region):
        '''
        Used internally for csv_dir and csv_filename.
        '''
        return 'region-{}-{}-{}-{}'.format(prediction_region.x1,
                                           prediction_region.x2,
                                           prediction_region.y1,
                                           prediction_region.y2)

    def __str__(self):
        return '{} {} {} {}'.format(self.region_metadata, self.clustering_metadata,
                                    self.distance_measure, self.model_params)
