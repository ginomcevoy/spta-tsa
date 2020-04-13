'''
Light Weight K-Medoids Implementation
Based on https://github.com/shenxudeu/K_Medoids/blob/master/k_medoids.py
and https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
'''

import logging
import numpy as np
from copy import deepcopy

from numpy.random import choice
from numpy.random import seed

from . import Medoid, get_medoid_indices

logger = logging.getLogger()


def initial_medoids(X, k, random_seed):
    seed(random_seed)
    samples = choice(len(X), size=k, replace=False)

    logger.info('Initial medoids indices: {}'.format(str(samples)))
    medoids_data = zip(samples, X[samples, :])  # (index0, series0), (index1, series1), ...
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
            for i in range(len(X)):
                if i == medoid_j.index:
                    dist_mat[i, j] = 0.
                else:
                    dist_mat[i, j] = distance_measure.measure(X[i, :], medoid_j.series)

    mask = np.argmin(dist_mat, axis=1)
    members = np.zeros(len(X), dtype=np.int8)
    costs = np.zeros(k)
    for i in range(0, k):
        mem_id = np.where(mask == i)
        members[mem_id] = i
        costs[i] = distance_measure.combine(dist_mat[mem_id, i])

    total_cost = distance_measure.combine(costs)
    return members, costs, total_cost, dist_mat


def run_kmedoids(X, k, distance_measure, seed=1, max_iter=1000, tol=0.001, verbose=True):
    '''run algorithm return centers, members, and etc.'''
    # Get initial centers
    n_samples, n_features = X.shape
    medoids = initial_medoids(X, k, seed)

    members, costs, tot_cost, dist_mat = _get_cost(X, medoids, distance_measure)
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

                members_, costs_, tot_cost_, dist_mat_ = _get_cost(X, medoids_,
                                                                   distance_measure)

                if tot_cost - tot_cost_ > tol:
                    members, costs, tot_cost, dist_mat = members_, costs_, tot_cost_, dist_mat_
                    medoids = medoids_
                    SWAPPED = True
                    medoid_indices = get_medoid_indices(medoids)
                    if verbose:
                        logger.debug('Change medoids to {}'.format(str(medoid_indices)))

        if cc > max_iter:
            if verbose:
                logger.info('End Searching by reaching maximum iteration: {}'.format(max_iter))
            break
        if not SWAPPED:
            if verbose:
                logger.info('End Searching by no swaps')
            break
        cc += 1

    return medoids, members, costs, tot_cost, dist_mat
