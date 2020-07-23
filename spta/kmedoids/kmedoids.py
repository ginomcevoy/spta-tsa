'''
Light Weight K-Medoids Implementation
Based on https://github.com/shenxudeu/K_Medoids/blob/master/k_medoids.py
and https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
'''

import logging
import numpy as np
from copy import deepcopy
from collections import namedtuple
from numpy.random import choice, seed
import time

from spta.distance.dtw import DistanceByDTW

from . import Medoid, get_medoid_indices

logger = logging.getLogger()

''' The metadata for a K-medoids run '''
KmedoidsMetadata = namedtuple('KmedoidsMetadata', ('k', 'distance_measure', 'initial_medoids',
                                                   'random_seed', 'mode', 'max_iter', 'tol',
                                                   'verbose'))

''' A K-mediods result '''
KmedoidsResult = namedtuple('KmedoidsResult', ('k', 'random_seed', 'mode', 'medoids', 'labels',
                                               'costs', 'total_cost', 'medoid_distances'))


def choose_initial_medoids(X, k, random_seed, initial_indices=None):
    seed(random_seed)

    if not initial_indices:
        initial_indices = choice(len(X), size=k, replace=False)

    logger.info('Initial medoids indices: {}'.format(str(initial_indices)))

    # (index0, series0), (index1, series1), ...
    medoids_data = zip(initial_indices, X[initial_indices, :])
    medoids = [
        Medoid(index, series)
        for index, series
        in medoids_data
    ]
    return medoids


def _get_cost(X, medoids, distance_measure):
    '''return total cost and cost of each cluster'''

    k = len(medoids)
    dist_mat = np.zeros((len(X), k))

    # check if the distance matrix has been precomputed
    if hasattr(distance_measure, 'distance_matrix') \
            and distance_measure.distance_matrix is not None:

        # reuse data from the complete distance matrix
        # we need the distances between each point in the region and each medoid
        # easiest way to get this is to get the distances at the medoids and transpose it
        medoid_indices = get_medoid_indices(medoids)
        dist_mat_tr = distance_measure.distance_matrix[medoid_indices, :]
        dist_mat = dist_mat_tr.transpose()

    else:
        # compute distance (sub)matrix for the medoids
        dist_mat = np.zeros((len(X), k))

        # iterate clusters
        for j in range(0, k):
            medoid_j = medoids[j]

            # iterate all points, and calculate the distance between each point and the medoid
            # of the current cluster
            for i in range(len(X)):
                if i == medoid_j.index:
                    dist_mat[i, j] = 0.
                else:
                    dist_mat[i, j] = distance_measure.measure(X[i, :], medoid_j.series)

    # this will find, for each point, the index that minimizes the distance of that point
    # to a medoid (1-d array, length same as X)
    # this effectively represents the membership of each point to a cluster
    labels = np.argmin(dist_mat, axis=1)

    # calculate intra-cluster cost for each cluster
    costs = np.zeros(k)
    for i in range(0, k):
        # the cost is the distance from each point to its cluster, use the label to find the
        # points in a cluster
        members_in_cluster_i = np.where(labels == i)
        costs[i] = np.sum(dist_mat[members_in_cluster_i, i])

    # total_cost = distance_measure.combine(costs)

    # compute the total cost, sum of each intra-cluster cost
    total_cost = np.sum(costs)

    return labels, costs, total_cost, dist_mat


def candidate_generator_for_lite_kmedoids(n_samples, labels, cluster_label):
    '''
    The "lite" k-medoids implementation looks for a better medoid among the current members of
    this cluster.
    '''
    cluster_indices = np.where(labels == cluster_label)[0]
    for index in cluster_indices:
        yield index


def candidate_generator_for_robust_kmedoids(n_samples, labels, cluster_label):
    '''
    The "robust" k-medoids implementation looks for a better medoid in all samples,
    even in other clusters.
    '''
    for index in range(0, n_samples):
        yield index


def run_kmedoids(X, k, distance_measure, initial_medoids=None, random_seed=1, mode='lite',
                 max_iter=1000, tol=0.001, verbose=True):
    '''run algorithm return centers, labels, and etc.'''

    start_time = time.time()

    # initial medoids
    n_samples, n_features = X.shape
    medoids = choose_initial_medoids(X, k, random_seed, initial_medoids)

    # assign initial members
    labels, costs, tot_cost, dist_mat = _get_cost(X, medoids, distance_measure)
    cc, SWAPPED = 0, True

    # robust or lite?
    if mode == 'robust':
        candidate_generator = candidate_generator_for_robust_kmedoids
    elif mode == 'lite':
        candidate_generator = candidate_generator_for_lite_kmedoids

    while True:

        # at the beginning of this loop, set SWAPPED flag to false
        # if the flag remains false after iterating all points and clusters,
        # then medoids provide a local minimum cost, and the algorithm can exit
        SWAPPED = False
        medoid_indices = get_medoid_indices(medoids)
        logger.debug('medoid indices at iteration {}: {}'.format(cc, medoid_indices))

        # iterate clusters: j from 0 to k
        for j in range(0, k):

            # look for better medoids that reduce the total cost (sum of distances) of the current
            # cluster. The search space is determined by the mode:
            # - The "robust" k-medoids implementation looks for a better medoid in all samples,
            #   even in other clusters.
            # - The "lite" k-medoids implementation looks for a better medoid among the current
            #   members of this cluster.

            for i in candidate_generator(n_samples, labels, j):

                if i in medoid_indices:
                    # this sample is already a medoid of a cluster, don't consider it
                    continue

                # consider a new medoid for cluster j
                medoids_ = deepcopy(medoids)
                medoids_[j] = Medoid(i, X[i])

                # assign members again, considering the new medoid
                # this gives a new total cost of the cluster
                labels_, costs_, tot_cost_, dist_mat_ = _get_cost(X, medoids_,
                                                                  distance_measure)

                # if the total cost has been reduced, then the list of medoids is 'better'
                # use these new medoids, save new costs
                if tot_cost - tot_cost_ > tol:
                    labels, costs, tot_cost, dist_mat = labels_, costs_, tot_cost_, dist_mat_
                    medoids = medoids_

                    # flag that indicates that the algorithm has found new medoids
                    # this means that the algorithm will continue to run
                    SWAPPED = True
                    medoid_indices = get_medoid_indices(medoids)
                    if verbose:
                        logger.debug('Change medoids to {}'.format(str(medoid_indices)))

        if cc > max_iter:
            if verbose:
                logger.info('End Searching by reaching maximum iteration: {}'.format(max_iter))
                logger.info('Final medoid indices: {}'.format(get_medoid_indices(medoids)))
            break
        if not SWAPPED:
            if verbose:
                logger.info('End Searching by no swaps')
                logger.info('Final medoid indices: {}'.format(get_medoid_indices(medoids)))
            break
        cc += 1

    elapsed_time = time.time() - start_time

    np.set_printoptions(precision=3)
    logger.debug('Intra-cluster costs: {}'.format(costs))

    result = KmedoidsResult(k, random_seed, mode, get_medoid_indices(medoids), labels, costs,
                            tot_cost, dist_mat)

    if verbose:
        show_report(result, elapsed_time)

    return result


def show_report(result, elapsed_time):

    medoids = result.medoids

    logger.info('-------------------------------------------')
    logger.info('K-medoids for k={}, seed={}, mode={}'.format(result.k, result.random_seed,
                                                              result.mode))
    logger.info('-------------------------------------------')
    logger.info('Medoids={}'.format(medoids))
    logger.info('Sum of intra-cluster costs: {}'.format(result.total_cost))
    logger.info('Elapsed time: {:.2f}s'.format(elapsed_time))

    # calculate size of each cluster: count the number of members for each label
    _, points_per_cluster = np.unique(result.labels, return_counts=True)
    total_points = len(result.labels)

    for i in range(0, result.k):
        # show info per cluster
        points_i = points_per_cluster[i]
        coverage_i = points_i * 100.0 / total_points
        cluster_msg = 'Cluster {}: medoid={}, {} points ({:.1f}%)'
        logger.info(cluster_msg.format(i, medoids[i], points_i, coverage_i))


def kmedoids_default_metadata(k, distance_measure=DistanceByDTW(), initial_medoids=None,
                              random_seed=1, mode='lite', max_iter=1000, tol=0.001, verbose=True):
    '''
    Metadata for K-medoids with default values. Still needs a value for k.
    '''
    return KmedoidsMetadata(k=k, distance_measure=distance_measure,
                            initial_medoids=initial_medoids, random_seed=random_seed, mode=mode,
                            max_iter=max_iter, tol=tol, verbose=verbose)


def run_kmedoids_from_metadata(X, kmediods_metadata):
    return run_kmedoids(X, kmediods_metadata.k, kmediods_metadata.distance_measure,
                        kmediods_metadata.initial_medoids, kmediods_metadata.random_seed,
                        kmediods_metadata.mode, kmediods_metadata.max_iter, kmediods_metadata.tol,
                        kmediods_metadata.verbose)


def kmedoids_suite_metadata(k_values, seed_values, distance_measure=DistanceByDTW(), mode='lite',
                            initial_medoids=None, max_iter=1000, tol=0.001, verbose=True):
    '''
    Builds a list of KmedoidsMetadata based on the cartesian product of k_values and seed_values.
    Can override default values for other parameters.
    '''
    for k in k_values:
        for random_seed in seed_values:
            yield KmedoidsMetadata(k=k, distance_measure=distance_measure,
                                   initial_medoids=initial_medoids, random_seed=random_seed,
                                   mode=mode, max_iter=max_iter, tol=tol, verbose=verbose)


if __name__ == '__main__':
    from spta.region import Region
    from spta.region.metadata import SpatioTemporalRegionMetadata
    from spta.util import log as log_util

    log_util.setup_log('DEBUG')

    # Run k=2 on sp_small dataset
    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)
    nordeste_small = nordeste_small_md.create_instance()

    # load pre-computed distances
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(nordeste_small_md.distances_filename,
                                         nordeste_small_md.region)

    X = nordeste_small.as_2d

    k = 2
    metadata = kmedoids_default_metadata(k, distance_measure=distance_dtw)
    run_kmedoids_from_metadata(X, metadata)
