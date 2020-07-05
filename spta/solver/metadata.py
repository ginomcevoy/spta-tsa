class SolverMetadata(object):

    def __init__(self, region_metadata, clustering_metadata, distance_measure):
        self.region_metadata = region_metadata
        self.clustering_metadata = clustering_metadata
        self.distance_measure = distance_measure

    @property
    def solver_subdir(self):
        return '{!r}/{!r}/{!r}'.format(self.region_metadata,
                                       self.clustering_metadata,
                                       self.distance_measure)

    @property
    def csv_dir(self):
        return 'csv/{}'.format(self.solver_subdir)

    @property
    def pickle_dir(self):
        return 'pickle/{}'.format(self.solver_subdir)

    @property
    def plot_dir(self):
        return 'plots/{}'.format(self.solver_subdir)
