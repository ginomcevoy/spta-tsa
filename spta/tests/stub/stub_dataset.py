
from spta.dataset.base import FileDataset

from . import stub_region


class StubFileDataset(FileDataset):
    '''
    An implementation of FileDataset for automatic tests.
    It does not read from the filesystem, but it can write to it by receiving
    a (temporal) directory.

    The years passed as parameters are ignored for the slicing, but are used
    to create the cache filenames.
    '''

    def __init__(self, dataset_temporal_md, temp_dir):
        super(StubFileDataset, self).__init__(dataset_temporal_md)
        self.temp_dir = temp_dir

    def read_raw(self, series_start, series_end):
        '''
        Returns the dataset as a numpy multi-dimensional array.
        '''
        return stub_region.numpy_3d_4spd_stub()

    def cache_filename(self, temporal_md):
        return '{}/test_{!r}.npy'.format(self.temp_dir, temporal_md)
