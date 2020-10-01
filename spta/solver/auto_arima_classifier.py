'''
A solver that calculates the result of a prediction query using the outcome of an external
classifier (e.g. Neural Network). The classifier is a function that, given an input series of size
tp, a clustering suite and a region, returns one of the medoids from any of the clustering
metadata that are available in the clustering suite.

cs = { md1, md2, ... mdN }

classifier(s, cs, r) = f: (R^{tp}, CS, NxNxNxN) -> (md_i, m_j), where:

- s in R^p is a time series of size tp
- cs is one of the available clustering suites of the CS space
- r is a 2d rectangle specified by its 4 coordinates
- md_i in clustering_suite is a metadata for a clustering, e.g (k=10, seed=0, mode=lite)
  for k-medoids
- m_i in M is a medoid found by applying the partitioning specified by metadata_i on the region r.

Here, the following assumptions are made:

- Assume that the classifier has given us the (md_i, m_j) tuple for the series s in point P. The
  task of the solver is then to retrieve the ARIMA model of the medoid m_j, that was saved when
  training a solver using md_i and auto_arima.

- Assume that the appropriate solver has been previously trained and saved as a pickle object:
  if the ARIMA model is not found, an error is returned. Also the error needs to be available.

- We assume a constant size tf for the predicted series. The output is obtained calling the
  corresponding ARIMA model to make a prediction of size tf.

The inputs are as follows:
- region metadata
- distance_measure
- auto_arima metadata
- a region R for the prediction query
- for each point P in R, the output of the classifier, a list of (md_i, m_j) tuples.
- the desired prediction size tf.

The output is, for each point P in R:
- the predicted series of size tf
- the generalization error of the model

TODO if using sliding window, we also need to use an offset and model.predict() instead of
model.forecast(), in order to make an in-sample prediction.
'''

from .auto_arima import AutoARIMASolverPickler


def retrieve_model_for_solver(solver_metadata, cluster_index):
    '''
    Use an instance of SolverMetadataWithClustering to load a previously saved solver,
    and retrieve the saved ARIMA model at the medoid representing the cluster of the given index.
    '''

    # load the previously saved solver
    pickler = AutoARIMASolverPickler(solver_metadata)
    solver = pickler.load_solver()

    # find the medoid given cluster_index
    chosen_medoid = solver.partition.medoids[cluster_index]
    return solver.arima_model_region_training.value_at(chosen_medoid)


if __name__ == '__main__':
    from spta.arima import AutoArimaParams
    from spta.clustering.kmedoids import KmedoidsClusteringMetadata
    from spta.distance.dtw import DistanceByDTW
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.solver.metadata import SolverMetadataBuilder

    region_metadata = SpatioTemporalRegionMetadata('nordeste_small', Region(40, 50, 50, 60),
                                                   2015, 2015, 1, scaled=False)
    model_params = AutoArimaParams(1, 1, 3, 3, None, True)
    test_len = 8
    error_type = 'sMAPE'

    k = 2
    clustering_metadata = KmedoidsClusteringMetadata(k, random_seed=1, mode='lite')
    distance_measure = DistanceByDTW()

    builder = SolverMetadataBuilder(region_metadata, model_params, test_len, error_type). \
        with_clustering(clustering_metadata, distance_measure)

    solver_metadata = builder.build()

    arima_model = retrieve_model_for_solver(solver_metadata, 1)
    print(arima_model)
