import numpy as np
import os

from spta.util import log as log_util
from spta.util import module as module_util


class FileDataset(log_util.LoggerMixin):
    '''
    Represents an abstract raw dataset that is saved in the filesystem.
    Subclasses represent a concrete dataset.
    '''

    def __init__(self, dataset_temporal_md):
        '''
        Receives an instance of TemporalMetadata.
        Subclasses *must* have this parameter as first constructor parameter.
        '''
        self.dataset_temporal_md = dataset_temporal_md

    def read_raw(self, series_start, series_end):
        '''
        Returns the dataset as a numpy multi-dimensional array.
        Subclasses should indicate how to read data from filesystem.
        '''
        raise NotImplementedError

    def cache_filename(self, temporal_md):
        '''
        The path of a file containing a temporal slice as a numpy array.
        Subclasses should resolve how to use the TemporalMetadata instance,
        a good way is to use repr(temporal_md) and build the filename based on
        that string.
        '''
        raise NotImplementedError

    def save_to_cache(self, dataset_slice, temporal_md):
        '''
        Stores a previously loaded dataset interval as a numpy file.
        Returns the filename of the saved file.

        dataset_slice
            a subset of the entire dataset, by time interval.

        temporal_md
            an instance of TemporalMetadata that describes dataset_slice.
        '''
        cache_filename = self.cache_filename(temporal_md)
        np.save(cache_filename, dataset_slice)
        self.logger.info('Saved dataset slice: {}'.format(cache_filename))

        return cache_filename

    def try_cache(self, temporal_md):
        '''
        Tries to loads a previously saved dataset slice from its numpy file.
        Returns both the dataset slice and the filename of the saved numpy.
        If the data was not saved before, returns (None, filename) without error.

        temporal_md
            an instance of TemporalMetadata that describes the requested dataset slice.
        '''
        cache_filename = self.cache_filename(temporal_md)
        dataset_slice = None

        try:
            if os.path.exists(cache_filename):
                dataset_slice = np.load(cache_filename)
            else:
                self.logger.debug('Saved dataset slice not found: {}'.format(cache_filename))

        except Exception:
            self.logger.warn('Error loading dataset slice: {}'.format(cache_filename))

        return dataset_slice, cache_filename

    def retrieve(self, temporal_md):
        '''
        Returns a temporal slice of this dataset, described by the supplied metadata.
        Tries to read the data from the cache, and if it is not available it will:
            - read the raw dataset stored in the filesystem and slice it;
            - save the dataset slice into the cache for later use.
        '''

        # if this succeeds, we are done
        dataset_slice, cache_filename = self.try_cache(temporal_md)
        if dataset_slice is not None:
            self.logger.info('Using cached dataset slice: {}'.format(cache_filename))
            return dataset_slice

        # Cache failed, read the raw dataset to get the dataset slice for the time interval requested.
        # To do this, we use the original metadata of this dataset, which is aware of real time.
        (year_start_request, year_end_request) = (temporal_md.year_start, temporal_md.year_end)
        (series_start, series_end) = self.dataset_temporal_md.years_to_series_interval(
            year_start_request, year_end_request)

        # subclass implementation to get the temporal interval
        dataset_slice = self.read_raw(series_start, series_end)

        # temporal_md may request a conversion (e.g. spd=4 to spd=1),
        # here we perform this operation by comparing original and requested sample frequency.
        converted_dataset_slice = \
            self.dataset_temporal_md.convert_time_to_series(dataset_slice, temporal_md.time_to_series)

        # save this dataset slice to cache for performance
        self.save_to_cache(converted_dataset_slice, temporal_md)

        log_msg = 'Read dataset slice: {} -> {}'
        self.logger.info(log_msg.format(cache_filename, converted_dataset_slice.shape))

        return converted_dataset_slice


def create_dataset_instance(full_class_name, **kwargs):
    '''
    Dynamically creates an instance of a subclass of FileDataset, using the full class name
    (package.module.class) as a string parameter.
    Here we could pass a dataset_temporal_md parameter to the constructor, or any other.
    Note: the parent class FileDataset needs a temporal metadata parameter!
    '''
    instance = module_util.createInstanceWithArgs(full_class_name, **kwargs)
    if not isinstance(instance, FileDataset):
        raise ValueError('Should represent a FileDataset: {}'.format(full_class_name))

    return instance
