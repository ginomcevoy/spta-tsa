# based on https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
# adapted for custom distance function

from copy import deepcopy
import numpy as np
from numpy.random import choice
from numpy.random import seed

from collections import namedtuple

Medoid = namedtuple('Medoid', ('index', 'series'))


def init_medoids(X, k):
    seed(1)
    samples = choice(len(X), size=k, replace=False)

    print('Initial medoids indices: ', samples)
    medoids_data = zip(samples, X[samples, :])  # (index0, series0), (index1, series1), ...
    medoids = [
        Medoid(index, series)
        for index, series
        in medoids_data
    ]
    return medoids

# medoids_initial = init_medoids(datapoints, 3)


def compute_cluster_distances(X, medoids, distance_measure):
    m = len(X)
    # medoids_shape = medoids.shape

    # If a 1-D array is provided,
    # it will be reshaped to a single row 2-D array
    # if len(medoids_shape) == 1:
    #    medoids = medoids.reshape((1, len(medoids)))
    k = len(medoids)

    S = np.empty((m, k))

    for i in range(m):
        # d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
        # S[i, :] = d_i**p

        distances_to_each_medoid = list()
        for j in range(0, k):
            medoid_j = medoids[j]
            # print('medoid_j {}'.format(medoid_j))
            distance_to_medoid_j = distance_measure.measure(X[i, :], medoid_j.series)
            distances_to_each_medoid.append(distance_to_medoid_j)

        S[i, :] = np.array(distances_to_each_medoid)

    # print('computed distances for k={} medoids: {}'.format(k, S))
    return S

# S = compute_cluster_distances(datapoints, medoids_initial, 2)


def assign_labels(S):
    return np.argmin(S, axis=1)


# labels = assign_labels(S)

def update_medoids(X, medoids, distance_measure):

    S = compute_cluster_distances(X, medoids, distance_measure)
    labels = assign_labels(S)
    print('got updated labels ', labels)

    k = len(medoids)
    out_medoids = medoids

    # for i in set(labels):
    for i in range(0, k):

        distances = compute_cluster_distances(X, [medoids[i], ], distance_measure)
        avg_dissimilarity = distance_measure.combine(distances)

        # cluster_points = X[labels == i]

        # instead of iterating over a cluster, iterate over all series and then skip
        # points not in the cluster, this way we know the index
        for (index, datap) in enumerate(X):

            # don't evaluate points outside of the cluster
            # also don't evaluate the medoid itself
            if labels[index] != i or index == medoids[i].index:
                continue

            print('looking at index {} for medoid {}'.format(index, i))

            new_medoid = Medoid(index, datap)
            # print('new medoid: {}'.format(new_medoid))
            new_distances = compute_cluster_distances(X, [new_medoid, ], distance_measure)
            new_dissimilarity = distance_measure.combine(new_distances)

            if new_dissimilarity < avg_dissimilarity:
                avg_dissimilarity = new_dissimilarity

                out_medoids[i] = new_medoid
                print('updated medoid {} -> {}'.format(i, index))

                S = compute_cluster_distances(X, medoids, distance_measure)
                labels = assign_labels(S)
                print('computed distances for k={} medoids: {}'.format(k, S))
                print('updated labels from change: ', labels)

    return out_medoids


def has_converged(old_medoids, medoids):
    # return set([tuple(x.series) for x in old_medoids]) == set([tuple(x.series) for x in medoids])
    return [x.index for x in old_medoids] == [x.index for x in medoids]


# Full algorithm
def kmedoids(X, k, distance_measure, starting_medoids=None, max_steps=np.inf):
    if starting_medoids is None:
        medoids = init_medoids(X, k)
    else:
        medoids = starting_medoids

    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids = deepcopy(medoids)

        S = compute_cluster_distances(X, medoids, distance_measure)

        labels = assign_labels(S)
        print('got labels ', labels)

        new_medoids = update_medoids(X, medoids, distance_measure)

        converged = has_converged(old_medoids, new_medoids)
        if converged:
            print('converged!')
        medoids = new_medoids
        i += 1

    return (medoids, labels)
