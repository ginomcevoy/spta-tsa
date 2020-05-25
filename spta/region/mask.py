import numpy as np

from .base import BaseRegion


class MaskRegion(BaseRegion):
    '''
    A 2-d region that is used as a mask to indicate clustering of a spatial or spatio-temporal
    region. It remains basic enough so that only mask-specific functions can applied to it.
    The cluster must be identified by a label. For a clustering with k clusters, it is assumed
    that the label is an integer between 0 and k-1.

    The numpy_dataset remains opaque here, subclasses should determine the nature of the dataset.
    Since this is a region, x_len and y_len are available as second-to-last and last dimensions.

    There are two main functionalities of the mask:
    1. Indicate whether a point in the region belongs to the cluster (is_member function)
    2. Iterate over the points in the cluster, using the functionality in 1. in the iterator.
    '''

    def __init__(self, numpy_dataset, label):
        super(MaskRegion, self).__init__(numpy_dataset)
        self.label = label

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
    coordinates. The value indicates the label of the cluster to which the point belongs to.

    All clusters created by a clustering algorithm will get the same MaskRegion, except for the
    label value.
    '''

    def __init__(self, numpy_dataset, label):
        # assume that the numpy_dataset is a 2-d array
        assert numpy_dataset.ndim == 2

        super(MaskRegionCrisp, self).__init__(numpy_dataset, label)

        # save number of members in cluster
        self.cluster_len = np.count_nonzero(self.numpy_dataset == label)

    def is_member(self, point):
        '''
        Returns True iff the point is a member of the cluster with this mask.
        Implemented by checking the label value in the 2-d region.
        '''
        # sanity check
        if point is None:
            return False

        return self.numpy_dataset[point.x, point.y] == self.label

    def clone(self):
        return MaskRegionCrisp(np.copy(self.numpy_dataset), self.label)

    @classmethod
    def from_1d_labels(cls, labels, label, x_len, y_len):
        '''
        Creates an instance of MaskRegionCrisp using a 1-d label array as input.
        Requires the region shape.
        '''
        label_2d = labels.reshape(x_len, y_len)
        return MaskRegionCrisp(label_2d, label)
