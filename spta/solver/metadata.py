import os


class SolverMetadata(object):

    def __init__(self, region_metadata, clustering_metadata, distance_measure, model_params):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure
        self.model_params = model_params

    def output_dir(self, output_prefix):
        output_dir_clustering = self.clustering_metadata.output_dir(output_prefix,
                                                                    self.region_metadata,
                                                                    self.distance_measure)
        model_params_dir = '{!r}'.format(self.model_params)
        return os.path.join(output_dir_clustering, model_params_dir)

    def pickle_dir(self):
        return self.clustering_metadata.pickle_dir(self.region_metadata, self.distance_measure)

    def __str__(self):
        return '{} {} {} {}'.format(self.region_metadata, self.clustering_metadata,
                                    self.distance_measure, self.model_params)
