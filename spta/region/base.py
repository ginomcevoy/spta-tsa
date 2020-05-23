import numpy as np

from . import Point
import spta.region.spatial
from spta.util import log as log_util


class BaseRegion(log_util.LoggerMixin):
    '''
    The base class for DomainRegion and MaskRegion.

    It wraps a multi-dimensional numpy array to provide some basic additional behavior.
    Has an internal logger instance via LoggerMixin.
    '''

    def __init__(self, numpy_dataset, **kwargs):
        super(BaseRegion, self).__init__()
        self.numpy_dataset = numpy_dataset

        # used for iterating over points
        self.point_index = 0

        # convenience
        self.shape = numpy_dataset.shape
        self.ndim = numpy_dataset.ndim

        # makes iterations faster
        # this implemetation can be used for both spatial regions (x, y) and spatio-temporal
        # regions with (series, x, y) shape
        self.x_len = self.shape[self.ndim - 2]
        self.y_len = self.shape[self.ndim - 1]

    @property
    def as_numpy(self):
        return self.numpy_dataset

    def empty_region_2d(self):
        '''
        Returns an empty SpatialRegion with the same shape as this region.
        '''
        empty_region_np = np.empty((self.x_len, self.y_len))
        return spta.region.spatial.SpatialRegion(empty_region_np)

    def save(self, filename):
        '''
        Saves dataset to a file.
        '''
        ds_numpy = self.as_numpy
        np.save(filename, ds_numpy)
        self.logger.info('Saved to {}: {}'.format(filename, ds_numpy.shape))

    def __iter__(self):
        '''
        Used for iterating over points
        '''
        return self

    def __next__(self):
        '''
        This iterator will iterate over points. It can be reused by subclasses to iterate
        over (point, value) or (point, series) tuples.
        '''

        # the index will iterate from Point(0, 0) to Point(x_len - 1, y_len - 1)
        if self.point_index >= self.y_len * self.x_len:
            # reached the end, no more points to iterate
            # stop iteration, but allow reuse of iterator from start again
            self.point_index = 0
            raise StopIteration

        # iterate, prepare for next item
        i = int(self.point_index / self.y_len)
        j = self.point_index % self.y_len
        self.point_index += 1

        # return current point in iteration
        return Point(i, j)

    def __str__(self):
        '''
        A string representation: if a name is available, return it.
        Otherwise, return a generic name.
        '''
        if hasattr(self, 'name'):
            return self.name
        else:
            return 'Region {}'.format(self.shape)
