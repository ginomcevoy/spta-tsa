import logging
import numpy as np

from . import Region, reshape_1d_to_2d


# model + test + forecast -> error

# arimitas -> error de cada arima en su entrenamiento
# 1 model + test/region + forecast/region -> error en cada punto

class SpatialRegion:

    def __init__(self, numpy_dataset):
        self.numpy_dataset = numpy_dataset
        self.log = logging.getLogger()

    @property
    def as_numpy(self):
        return self.numpy_dataset

    @property
    def shape(self):
        return self.numpy_dataset.shape

    @property
    def ndim(self):
        return self.numpy_dataset.ndim

    def region_subset(self, region):
        '''
        region: Region namedtuple
        '''
        numpy_region_subset = self.numpy_dataset[region.x1:region.x2, region.y1:region.y2]
        return SpatialRegion(numpy_region_subset)

    def value_at(self, point):
        return self.numpy_dataset[point.x, point.y]

    @property
    def as_array(self):
        '''
        Goes from 2d to 1d
        '''
        (x_len, y_len) = self.numpy_dataset.shape
        return self.numpy_dataset.reshape(x_len * y_len)

    def save(self, filename):
        '''
        Saves dataset to a file.
        '''
        ds_numpy = self.as_numpy
        np.save(filename, ds_numpy)
        self.log.info('Saved to {}: {}'.format(filename, ds_numpy.shape))

    @classmethod
    def create_from_1d(cls, list_1d, x, y):
        '''
        Given a list, reshapes the elements to form a 2d region with (x, y) shape.
        '''
        numpy_dataset = reshape_1d_to_2d(list_1d, x, y)
        return SpatialRegion(numpy_dataset)

    @classmethod
    def region_with_zeroes(cls, region):
        '''
        Creates a spatial region filled with zeroes, with the dimensions given by the region.
        Notice that indices will start at (0, 0)
        '''
        (x_len, y_len) = (Region.x2 - Region.x1, Region.y2 - Region.y1)
        zero_array = np.zeros((x_len, y_len))
        return SpatialRegion(zero_array)

    def __str__(self):
        return str(self.numpy_dataset)
