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

from spta.distance.dtw import DistanceByDTW

from . import Medoid, get_medoid_indices

logger = logging.getLogger()

''' The metadata for a K-medoids run '''
KmedoidsMetadata = namedtuple('KmedoidsMetadata', ('k', 'distance_measure', 'initial_medoids',
                                                   'random_seed', 'max_iter', 'tol', 'verbose'))

''' A K-mediods result '''
KmedoidsResult = namedtuple('KmedoidsResult', ('k', 'random_seed', 'medoids', 'labels', 'costs',
                                               'total_cost', 'medoid_distances'))


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
        # compute distance matrix for the medoids
        dist_mat = np.zeros((len(X), k))

        for j in range(0, k):

            medoid_j = medoids[j]
            # print('Evaluating medoid at {}...'.format(j))
            for i in range(len(X)):
                if i == medoid_j.index:
                    dist_mat[i, j] = 0.
                else:
                    dist_mat[i, j] = distance_measure.measure(X[i, :], medoid_j.series)

    # print(dist_mat)
    mask = np.argmin(dist_mat, axis=1)
    labels = np.zeros(len(X), dtype=np.int8)
    costs = np.zeros(k)
    for i in range(0, k):
        mem_id = np.where(mask == i)
        labels[mem_id] = i
        # costs[i] = distance_measure.combine(dist_mat[mem_id, i])
        costs[i] = np.sum(dist_mat[mem_id, i])

    total_cost = distance_measure.combine(costs)
    return labels, costs, total_cost, dist_mat


def run_kmedoids(X, k, distance_measure, initial_medoids=None, random_seed=1, max_iter=1000,
                 tol=0.001, verbose=True):
    '''run algorithm return centers, labels, and etc.'''
    # Get initial centers
    n_samples, n_features = X.shape
    medoids = choose_initial_medoids(X, k, random_seed, initial_medoids)

    labels, costs, tot_cost, dist_mat = _get_cost(X, medoids, distance_measure)
    cc, SWAPPED = 0, True

    while True:
        SWAPPED = False
        medoid_indices = get_medoid_indices(medoids)
        logger.debug('medoid indices {}'.format(medoid_indices))
        for i in range(0, n_samples):

            for j in range(0, k):

                if i in medoid_indices:
                    continue

                medoids_ = deepcopy(medoids)
                medoids_[j] = Medoid(i, X[i])

                labels_, costs_, tot_cost_, dist_mat_ = _get_cost(X, medoids_,
                                                                  distance_measure)

                # logger.debug('Cost improvement: {}'.format(tot_cost - tot_cost_))
                if tot_cost - tot_cost_ > tol:
                    labels, costs, tot_cost, dist_mat = labels_, costs_, tot_cost_, dist_mat_
                    medoids = medoids_
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

    result = KmedoidsResult(k, random_seed, medoids, labels, costs, tot_cost, dist_mat)
    show_report(result)

    return result


def show_report(result):

    medoids = get_medoid_indices(result.medoids)

    logger.info('----------------------------')
    logger.info('K-medoids for k={}, seed={}'.format(result.k, result.random_seed))
    logger.info('----------------------------')
    logger.info('Medoids={}'.format(medoids))

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
                              random_seed=1, max_iter=1000, tol=0.001, verbose=True):
    '''
    Metadata for K-medoids with default values. Still needs a value for k.
    '''
    return KmedoidsMetadata(k=k, distance_measure=distance_measure,
                            initial_medoids=initial_medoids, random_seed=random_seed,
                            max_iter=max_iter, tol=tol, verbose=verbose)


def run_kmedoids_from_metadata(X, kmediods_metadata):
    return run_kmedoids(X, kmediods_metadata.k, kmediods_metadata.distance_measure,
                        kmediods_metadata.initial_medoids, kmediods_metadata.random_seed,
                        kmediods_metadata.max_iter, kmediods_metadata.tol,
                        kmediods_metadata.verbose)
