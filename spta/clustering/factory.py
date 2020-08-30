from .regular import RegularClusteringMetadata, RegularClusteringAlgorithm
from .kmedoids import KmedoidsClusteringMetadata, KmedoidsClusteringAlgorithm


class ClusteringMetadataFactory:
    '''
    Create an instance of ClusteringMetadata with the correct subtype.
    '''

    def instance(self, name, k, **params):
        '''
        Create an instance of a ClusteringMetadata subtype given its parameters
        '''
        if name == 'regular':
            # regular only has k
            return RegularClusteringMetadata(k)

        if name == 'kmedoids':
            # kmedoids has k, random_seed and many other optional parameters
            random_seed = params.pop('random_seed')
            return KmedoidsClusteringMetadata(k=k, random_seed=random_seed, **params)

        raise ValueError('Invalid name of clustering metadata: {}'.format(name))

    def from_repr(self, repr_string):
        '''
        Create an instance of a ClusteringMetadata subtype given its string representation
        '''
        # the first part indicates the name (type)
        parts = repr_string.split('_')

        if parts[0] == 'regular':
            return RegularClusteringMetadata.from_repr(repr_string)

        if parts[0] == 'kmedoids':
            return KmedoidsClusteringMetadata.from_repr(repr_string)

        raise ValueError('Invalid representation of clustering metadata: {}'.format(repr_string))

class ClusteringFactory:

    def __init__(self, distance_measure):
        self.distance_measure = distance_measure

    def instance(self, metadata):
        if metadata.name == 'regular':
            return RegularClusteringAlgorithm(metadata, self.distance_measure)

        if metadata.name == 'kmedoids':
            return KmedoidsClusteringAlgorithm(metadata, self.distance_measure)

        raise ValueError('clustering metadata not recognized: {}'.format(metadata))
