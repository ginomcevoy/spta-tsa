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

from spta.region.mask import MaskRegionFuzzy
from spta.distance.dtw import DistanceByDTW

from . import Medoid, get_medoid_indices

logger = logging.getLogger()


class KmedoidsFuzzy(object):
    '''
    Run K-medoids fuzzy based on:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.41.2622&rep=rep1&type=pdf

    Uses by default the "fuzzifier" implementation from FCM, a probabilistic approach.
    '''

    def __init__(self):
        pass


''' The parameters for a K-medoids run '''
KmedoidsFuzzyParams = namedtuple('KmedoidsFuzzyParams', ('k', 'm', 'distance_measure',
                                                         'initial_medoids', 'random_seed',
                                                         'max_iter', 'tol', 'verbose'))

''' A K-mediods fuzzy result '''
KmedoidsFuzzyResult = namedtuple('KmedoidsFuzzyResult', ('k', 'm', 'random_seed', 'medoids', 'uij',
                                                         'costs', 'total_cost'))


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


def compute_membership_fcm(X, m, medoids, distance_measure, tol):
    '''
    Given a distance measure and a list of medoids, find the membership of each point.
    The membership for each point i is a vector u such that uij is the degree of membership of
    point i to cluster j.

    Using FCM:

        uij = { [ (1/r(xj, vi))^(1/m-1) ] / sum([ (1/r(xj, vi))^(1/m-1) ], k=1, k=c) }

    This is a probabilistic approach:

        uij in [0, 1]
        0 < sum(uij, j=0, j=k-1) < N for all i
        sum(uij, i=0, i=n) = 1 for all j


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
        for i in range(0, k):
            medoid_i = medoids[i]

            # iterate all points, and calculate the distance between each point and the medoid
            # of the current cluster
            for j in range(len(X)):
                if j == medoid_i.index:
                    dist_mat[j, i] = 0.
                else:
                    dist_mat[j, i] = distance_measure.measure(X[j, :], medoid_i.series)

    uij = np.zeros((len(X), k))

    # the denominator of uij, the end value is the same for all clusters,
    # so there are len(X) values
    uij_den = np.zeros((len(X),))

    for i in range(0, k):
        r_xj_vi = dist_mat[:, i]
        # print('medoid: {}'.format(medoids[i].index))
        # print('r_xj_vi for i={}: {}'.format(i, r_xj_vi))

        # (1/r(xj, vi))^(1/m-1)
        #
        # use np.power(x1, x2)
        # x2 are the exponents. If x1.shape != x2.shape, they must be broadcastable to a
        # common shape (which becomes the shape of the output).
        uij[:, i] = np.power(1 / r_xj_vi, (1 / (m - 1)))
        # print('r_xj_vi_pwr', uij[:, i])

        # accumulate the sum in the denominator
        uij_den += uij[:, i]
        # print('uij_den', uij_den)

    for i in range(0, k):
        # divide by the denominator
        uij[:, i] = uij[:, i] / uij_den

        # special case for medoids (i = medoid index):
        # uij is (inf / inf) when j is the medoid's cluster, 0 for other clusters
        # 0 is calculated correctly, but (inf / inf) returns nan.
        # Here we set that value to 1 to indicate that the medoid has full membership
        # to its cluster
        uij[medoids[i].index, i] = 1

    # sanity check: adding all the membership values of a point should return 1 or very close to it
    np.testing.assert_allclose(np.sum(uij, axis=1), np.ones(len(X)), rtol=tol)

    # compute the cost of each cluster. This is the sum of the weighted distances of each sample
    # to its corresponding medoid, where the weights are uij
    fuzzy_costs = np.zeros(k)
    for i in range(0, k):
        fuzzy_costs[i] = np.sum(uij[:, i] * dist_mat[:, i])
    # print('fuzzy costs', fuzzy_costs)

    # compute the total cost, sum of each intra-cluster cost
    total_cost = np.sum(fuzzy_costs)
    return uij, fuzzy_costs, total_cost, dist_mat


def run_kmedoids_fuzzy(X, k, m, distance_measure, initial_medoids=None, random_seed=1,
                       max_iter=1000, tol=0.001, verbose=True):
    '''run algorithm return centers, labels, and etc.'''

    # initial medoids
    n_samples, n_features = X.shape
    medoids = choose_initial_medoids(X, k, random_seed, initial_medoids)

    # assign initial members
    labels, costs, tot_cost, dist_mat = compute_membership_fcm(X, m, medoids, distance_measure,
                                                               tol)
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
                uij_, costs_, tot_cost_, _ = compute_membership_fcm(X, m, medoids_,
                                                                    distance_measure, tol)

                # if the total cost has been reduced, then the list of medoids is 'better'
                # use these new medoids, save new costs
                if tot_cost - tot_cost_ > tol:
                    uij, costs, tot_cost, _ = uij_, costs_, tot_cost_, _
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

    result = KmedoidsFuzzyResult(k, m, random_seed, medoids, uij, costs, tot_cost)
    show_report(result)

    return result


def show_report(result):

    medoids = get_medoid_indices(result.medoids)

    logger.info('----------------------------')
    logger.info('K-medoids for k={}, seed={}'.format(result.k, result.random_seed))
    logger.info('----------------------------')
    logger.info('Medoids={}'.format(medoids))

    # total samples
    total_points = result.uij.shape[0]
    # print(result.uij)

    for i in range(0, result.k):
        # show info per cluster

        # To get number of points of each cluster, create fake fuzzy mask regions (1 x tot_points)
        mask_region_i = MaskRegionFuzzy.from_uij_and_region(result.uij, 1, total_points,
                                                            cluster_index=i, threshold=0)

        points_i = mask_region_i.cluster_len
        coverage_i = points_i * 100.0 / total_points
        cluster_msg = 'Cluster {}: medoid={}, with threshold=0 -> {} points ({:.1f}%)'
        logger.info(cluster_msg.format(i, medoids[i], points_i, coverage_i))


def kmedoids_fuzzy_default_params(k, m=2, distance_measure=DistanceByDTW(), initial_medoids=None,
                                  random_seed=1, max_iter=1000, tol=0.001, verbose=True):
    '''
    Default parameters for K-medoids fuzzy. Still needs a value for k.
    '''
    return KmedoidsFuzzyParams(k=k, m=m, distance_measure=distance_measure,
                               initial_medoids=initial_medoids, random_seed=random_seed,
                               max_iter=max_iter, tol=tol, verbose=verbose)


def run_kmedoids_fuzzy_from_params(X, kfuzzy_params):
    return run_kmedoids_fuzzy(X, kfuzzy_params.k, kfuzzy_params.m, kfuzzy_params.distance_measure,
                              kfuzzy_params.initial_medoids, kfuzzy_params.random_seed,
                              kfuzzy_params.max_iter, kfuzzy_params.tol, kfuzzy_params.verbose)


if __name__ == '__main__':
    from spta.region import Region
    from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata
    from spta.util import log as log_util

    log_util.setup_log('DEBUG')

    # Run k=2 on sp_small dataset
    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)
    nordeste_small = SpatioTemporalRegion.from_metadata(nordeste_small_md)

    # load pre-computed distances
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(nordeste_small_md.distances_filename,
                                         nordeste_small_md.region)

    X = nordeste_small.as_2d

    k = 2
    kfuzzy_params = kmedoids_fuzzy_default_params(k, m=2, distance_measure=distance_dtw)
    run_kmedoids_fuzzy_from_params(X, kfuzzy_params)
