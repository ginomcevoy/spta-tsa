from .regular import RegularClusteringAlgorithm
from .kmedoids import KmedoidsClusteringAlgorithm

class ClusteringFactory:

    def __init__(self, distance_measure):
        self.distance_measure = distance_measure

    def instance(self, metadata):
        if metadata.name == 'regular':
            return RegularClusteringAlgorithm(metadata, self.distance_measure)

        if metadata.name == 'kmedoids':
            return KmedoidsClusteringAlgorithm(metadata, self.distance_measure)

        raise ValueError('clustering metadata not recognized: {}'.format(metadata))
