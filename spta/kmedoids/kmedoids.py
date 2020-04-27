'''
Light Weight K-Medoids Implementation
Based on https://github.com/shenxudeu/K_Medoids/blob/master/k_medoids.py
and https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
'''

import logging
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from numpy.random import choice
from numpy.random import seed

from spta.region.spatial import SpatialRegion
from spta.util import plot as plot_util

from . import Medoid, get_medoid_indices

logger = logging.getLogger()


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
    members = np.zeros(len(X), dtype=np.int8)
    costs = np.zeros(k)
    for i in range(0, k):
        mem_id = np.where(mask == i)
        members[mem_id] = i
        # costs[i] = distance_measure.combine(dist_mat[mem_id, i])
        costs[i] = np.sum(dist_mat[mem_id, i])

    total_cost = distance_measure.combine(costs)
    return members, costs, total_cost, dist_mat


def run_kmedoids(X, k, distance_measure, initial_medoids=None, seed=1, max_iter=1000, tol=0.001,
                 verbose=True):
    '''run algorithm return centers, members, and etc.'''
    # Get initial centers
    n_samples, n_features = X.shape
    medoids = choose_initial_medoids(X, k, seed, initial_medoids)

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

                # logger.debug('Cost improvement: {}'.format(tot_cost - tot_cost_))
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
                logger.info('Final medoid indices: {}'.format(get_medoid_indices(medoids)))
            break
        if not SWAPPED:
            if verbose:
                logger.info('End Searching by no swaps')
                logger.info('Final medoid indices: {}'.format(get_medoid_indices(medoids)))
            break
        cc += 1

    return medoids, members, costs, tot_cost, dist_mat


def silhouette_spt(ks, spt_region, distance_measure, initial_medoids=None, seeds=(1,),
                   max_iter=1000, tol=0.001, verbose=True, show_graphs=True,
                   save_graphs='plots/silhouette'):
    '''
    Given a spatio-temporal region, creates silhouette graphs and calculates the silhouette
    average for each provided k, using k-medoids algorithm and provided distance function.
    It requires the distance_measure to have a distance matrix available, so this function will
    compute it if not provided.
    '''

    best_silhouette_avg = -1
    best_k = 0
    best_seed = None
    best_medoids = None
    best_labels = None

    # k-medoids expectes a matrix (n_samples x n_features)
    # this converts spatio-temporal region in a list of temporal series
    series_group = spt_region.as_2d

    # we also need the shape for graphs
    (_, x_len, y_len) = spt_region.shape

    # need a distance matrix for silhouette analysis
    if distance_measure.distance_matrix is None:
        distance_matrix = distance_measure.compute_distance_matrix(spt_region)
    distance_matrix = distance_measure.distance_matrix

    for a_seed in seeds:
        logger.info('Using seed: {}'.format(str(a_seed)))

        for k in ks:

            k_initial_medoids = None
            if initial_medoids:
                k_initial_medoids = initial_medoids[0:k]

            # apply k-medoids on the data using distance function
            kmedoids_result = run_kmedoids(series_group, k, distance_measure,
                                           initial_medoids=k_initial_medoids, seed=a_seed,
                                           max_iter=max_iter, tol=tol, verbose=verbose)
            (medoids, labels, costs, _, _) = kmedoids_result
            logger.debug(labels)
            logger.debug(costs)

            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with k={}, seed={}".format(k, a_seed)), fontsize=14, fontweight='bold')

            # plot the labels in 2d
            if show_graphs:
                label_region = SpatialRegion.create_from_1d(labels, x_len, y_len)
                plot_util.plot_discrete_spatial_region(label_region, 'Output mask', subplot=ax1)

            # build the silhouette graph, requires all distances
            silhouette_avg = plot_util.plot_clustering_silhouette(distance_matrix, labels,
                                                                  subplot=ax2)

            if show_graphs:
                plt.show()

            if save_graphs is not None:
                # save the figure for this k
                filename_k = '{}_k{}_seed{}.eps'.format(save_graphs, str(k), str(a_seed))
                fig.savefig(filename_k)

            # save best results
            if silhouette_avg > best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_k = k
                best_seed = a_seed
                best_medoids = medoids
                best_labels = labels

    return best_k, best_seed, best_medoids, best_labels
