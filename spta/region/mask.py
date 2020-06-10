import numpy as np

from spta.util import arrays as arrays_util
from .base import BaseRegion


class MaskRegion(BaseRegion):
    '''
    A 2-d region that is used as a mask to indicate clustering of a spatial or spatio-temporal
    region. It remains basic enough so that only mask-specific functions can applied to it.
    The cluster must be identified by an index (label). For a clustering with k clusters, it is
    assumed that the index is an integer between 0 and k-1.

    The numpy_dataset remains opaque here, subclasses should determine the nature of the dataset.
    Since this is a region, x_len and y_len are available as second-to-last and last dimensions.

    There are two main functionalities of the mask:
    1. Indicate whether a point in the region belongs to the cluster (is_member function)
    2. Iterate over the points in the cluster, using the functionality in 1. in the iterator.
    '''

    def __init__(self, numpy_dataset, cluster_index):
        super(MaskRegion, self).__init__(numpy_dataset)
        self.cluster_index = cluster_index

    def is_member(self, point):
        '''
        Returns True iff the point is a member of the cluster with this mask.
        Subclasses must override this.
        '''
        raise NotImplementedError

    def clone(self):
        '''
        Return an identical mask instance. This is useful because we get a new iterator index.
        Subclasses must override this.
        '''
        raise NotImplementedError

    def __next__(self):
        '''
        Iterate over points in the mask. Returns the next point that is a member of the current
        cluster.
        '''

        while True:

            # use the base iterator to get next candidate point in region
            # when the base iterator stops, we also stop
            try:
                candidate_point = super(MaskRegion, self).__next__()
            except StopIteration:
                # self.logger.debug('Base region iteration stopped')
                raise

            # self.logger.debug('{}: candidate = {}'.format(self, candidate_point))

            if self.is_member(candidate_point):
                # found a member of the cluster
                next_point = candidate_point
                break

            # the point was not in the mask, try with next candidate

        return next_point


class MaskRegionCrisp(MaskRegion):
    '''
    A mask region that is the result of applying a 'crisp' (non-fuzzy) clustering
    algorithm.

    It contains a 2-d array that indicates the membership of each point, using 2-d region
    coordinates. The value indicates the index of the cluster to which the point belongs to.

    All clusters created by a clustering algorithm can get a a MaskRegion with the same underlying
    dataset (membership matrix), but with indices ranging from 0 to k-1.
    '''

    def __init__(self, numpy_dataset, cluster_index):
        # assume that the numpy_dataset is a 2-d array (membership matrix)
        assert numpy_dataset.ndim == 2

        super(MaskRegionCrisp, self).__init__(numpy_dataset, cluster_index)

        # save number of members in cluster (constant for crisp clusters)
        self.cluster_len = np.count_nonzero(self.numpy_dataset == cluster_index)

    def is_member(self, point):
        '''
        Returns True iff the point is a member of the cluster with this mask.
        Implemented by checking the cluster index in the 2-d region.
        '''
        # sanity check
        if point is None:
            return False

        return self.numpy_dataset[point.x, point.y] == self.cluster_index

    def clone(self):
        return MaskRegionCrisp(np.copy(self.numpy_dataset), self.cluster_index)

    @classmethod
    def from_membership_array(cls, membership, cluster_index, x_len, y_len):
        '''
        Creates an instance of MaskRegionCrisp using a 1-d membership array as input.
        Requires the region shape.
        '''
        # reshape the array to get a membership matrix (shape of region)
        membership_2d = membership.reshape(x_len, y_len)
        return MaskRegionCrisp(membership_2d, cluster_index)

    @classmethod
    def with_regular_partition(cls, k, cluster_index, x_len, y_len):
        '''
        Creates an instance of MaskRegionCrisp based on the number of clusters, so that each
        cluster gets a rectangle of (approximately) the same size.
        The process is as follows:
            - Compute the most balanced divisors of k, e.g. 12 -> 4 x 3.
            - Partition the (x_len, y_len)  using these two values to create the cluster labels,
              e.g.
                  0  0  0  1  1  1 ...  3  3  3
                  0  0  0  1  1  1 ...  3  3  3
                  .....................
                  9  9  9 10 10 10 ..  11 11 11
                  9  9  9 10 10 10 ..  11 11 11

            - call from _membership_array with this membership matrix and corresponding
              cluster_index
        '''
        # all clusters in the partition will have this same membership matrix
        membership_regular = arrays_util.regular_partitioning(x_len, y_len, k)
        return cls.from_membership_array(membership_regular, cluster_index, x_len, y_len)


class MaskRegionFuzzy(MaskRegion):
    '''
    A mask region that is the result of applying a fuzzy clustering algorithm.
    It contains a 3-d array that indicates membership of each point, using 2-d region coordinates,
    The shape should be (k, x_len, y_len), where x_len, y_len determine the region and k is
    the number of clusters, with 0 <= cluster_index < k.

    For each coordinate with Point with index i, the array stored is uij, where j indicates the
    cluster index:

        At point P(x, y) with index i -> [ui0, ui1, ... uik]

    Membership is determined using a threshold value. The threshold indicates how dominant the
    highest value in uij is.

    If the threshold is 0, then this mask behaves like the 'crisp' version, each point belongs
    only to one cluster, the one with the highest uij.

    If threshold T is larger than 0, then this mask will indicate that the point with index i
    belongs to cluster j, if the following is satisfied:

        uim - uij <= T, where m is the index that maximizes uij at point i (best cluster)

    A limit threshold of 1 would mean that all points belong to all clusters.
    For values 0 <= T < 1, a medoid only belongs to its cluster (since uij = 1 for that cluster,
    0 for others)
    '''

    def __init__(self, numpy_dataset, cluster_index, threshold=0):

        # assume that the numpy_dataset is a 3-d array
        assert numpy_dataset.ndim == 3

        # sanity checks for threshold
        assert threshold >= 0
        assert threshold <= 1

        # sanity checks for cluster_index
        assert cluster_index >= 0
        assert cluster_index < numpy_dataset.shape[0]

        super(MaskRegionFuzzy, self).__init__(numpy_dataset, cluster_index)
        self.threshold = threshold

    @property
    def cluster_len(self):
        '''
        Calculates the cluster length (# of points) based on current threshold.
        '''
        # just iterate the points and return the length of the array
        points = [point for point in self]
        return len(points)

    def is_member(self, point):
        '''
        Returns True iff the point is can be considered a member of the cluster with this mask.
        Implemented by comparing the memberships to the threshold:

            uim - uij <= T, where m is the index that maximizes uij at point i (best cluster)
        '''
        # sanity check
        if point is None:
            return False

        # get the fuzzy membership at specified point
        u_point_j = self.numpy_dataset[:, point.x, point.y]

        # find the best cluster and its degree of membership
        best = np.argmax(u_point_j)
        u_point_best = u_point_j[best]

        # the membership for this cluster
        u_point_this = u_point_j[self.cluster_index]

        # apply the threshold to decide membership
        return u_point_best - u_point_this <= self.threshold

    def clone(self):
        return MaskRegionFuzzy(np.copy(self.numpy_dataset), self.cluster_index, self.threshold)

    @classmethod
    def from_uij_and_region(cls, uij, x_len, y_len, cluster_index, threshold):
        '''
        Creates an instance of MaskRegionFuzzy using a 2-d membership array and the region shape
        as input.
        '''
        # sanity check for input shape
        (N, k) = uij.shape
        assert N == x_len * y_len

        # MaskRegion reuses BaseRegion functionality that relies on [:, x, y] shape
        # uij has different ordering, fix this here
        # uij_swapped will have (k, N) shape
        uij_swapped = np.swapaxes(uij, 0, 1)

        # now reshape to mask desired input
        mask_np = uij_swapped.reshape((k, x_len, y_len))
        return MaskRegionFuzzy(mask_np, cluster_index, threshold)
