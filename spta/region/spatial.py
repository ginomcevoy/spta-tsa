import logging

from . import reshape_1d_to_2d


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

    @classmethod
    def create_from_1d(cls, list_1d, x, y):
        '''
        Given a list, reshapes the elements to form a 2d region with (x, y) shape.
        '''
        numpy_dataset = reshape_1d_to_2d(list_1d, x, y)
        return SpatialRegion(numpy_dataset)

    def __str__(self):
        return str(self.numpy_dataset)
