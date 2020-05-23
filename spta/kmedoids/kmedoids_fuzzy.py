'''
K-Medoids Fuzzy implementation
Based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.41.2622&rep=rep1&type=pdf
and https://github.com/shenxudeu/K_Medoids/blob/master/k_medoids.py
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
KmedoidsFuzzyMetadata = namedtuple('KmedoidsFuzzyMetadata', ('k', 'm', 'distance_measure',
                                                             'initial_medoids', 'random_seed',
                                                             'max_iter', 'tol', 'verbose'))

''' A K-mediods fuzzy result '''
KmedoidsFuzzyResult = namedtuple('KmedoidsFuzzyResult', ('k', 'm', 'random_seed', 'medoids',
                                                         'membership', 'costs', 'total_cost'))


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


def compute_membership_fcm(X, medoids, distance_measure):
    '''
    Given a distance measure and a list of medoids, find the membership of each point.
    The membership for each point i is a vector u such that uij is the degree of membership of
    point i to cluster j.

    Using FCM 

    This is a probabilistic approach:

        uij in [0, 1]
        0 < sum(uij, j=0, j=k-1) < N for all i
        sum(uij, )



    '''

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


def run_kmedoids(X, k, distance_measure, initial_medoids=None, random_seed=1, max_iter=1000,
                 tol=0.001, verbose=True):
    '''run algorithm return centers, labels, and etc.'''

    # initial medoids
    n_samples, n_features = X.shape
    medoids = choose_initial_medoids(X, k, random_seed, initial_medoids)

    # assign initial members
    labels, costs, tot_cost, dist_mat = _get_cost(X, medoids, distance_measure)
    cc, SWAPPED = 0, True

    while True:

        # at the beginning of this loop, set SWAPPED flag to false
        # if the flag remains false after iterating all points and clusters,
        # then medoids provide a local minimum cost, and the algorithm can exit
        SWAPPED = False
        medoid_indices = get_medoid_indices(medoids)
        logger.debug('medoid indices {}'.format(medoid_indices))

        # look for better medoids that reduce the total cost (sum of distances) of the clusters
        # this implementation looks for a better medoid in all samples, even in other clusters
        for i in range(0, n_samples):

            # iterate clusters: j from 0 to k
            for j in range(0, k):

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
