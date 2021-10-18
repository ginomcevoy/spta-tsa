import pandas as pd
import numpy as np

from .base import FileDataset
from .metadata import TemporalMetadata, AveragePentads


CACHE_FORMAT_2D = 'raw/chirps2d_{!r}.npy'
CACHE_FORMAT_3D = 'raw/chirps3d_{!r}.npy'

CSV_WITH_COORDS_AND_PENTADS = 'raw/data_cluster_distance_dtw_2010_2018.csv'


class DatasetCHIRPS2D(FileDataset):
    '''CHIRPS dataset, based on a numpy file that provides pentads in a 23604x1 region'''

    def __init__(self):
        '''Initialize the parent FileDataset with the metadata of this dataset.'''
        time_to_series = AveragePentads()
        dataset_temporal_md = TemporalMetadata(2010, 2018, time_to_series)
        super(DatasetCHIRPS2D, self).__init__(dataset_temporal_md)

    def cache_filename(self, temporal_md):
        '''
        The path of a file containing a temporal slice as a numpy array.
        '''
        return CACHE_FORMAT_2D.format(temporal_md)

    def retrieve(self, temporal_md):
        '''
        For now, we can only use the dataset if we already have a cache available,
        provided externally.
        '''
        dataset_avg_pentads, cache_filename = self.try_cache(temporal_md)
        if dataset_avg_pentads is None:
            raise ValueError('CHIRPS needs this file: {}'.format(cache_filename))

        log_msg = 'Using cached dataset of average pentads: {} -> {}'
        self.logger.info(log_msg.format(cache_filename, dataset_avg_pentads.shape))

        return dataset_avg_pentads


class DatasetCHIRPS3D(FileDataset):
    '''
    CHIRPS dataset, based on a CSV file that provides 23604 tuples for the pentads in the form
    (lat, lon, pent_1, ..., pent_73.
    '''

    def __init__(self):
        '''Initialize the parent FileDataset with the metadata of this dataset.'''
        time_to_series = AveragePentads()
        dataset_temporal_md = TemporalMetadata(2010, 2018, time_to_series)
        super(DatasetCHIRPS3D, self).__init__(dataset_temporal_md)

    def read_from_csv(self):
        '''
        Returns the dataset as a numpy multi-dimensional array.
        '''
        clusters_df = pd.read_csv(CSV_WITH_COORDS_AND_PENTADS)
        self.logger.debug('Read CSV: {}'.format(CSV_WITH_COORDS_AND_PENTADS))
        self.logger.debug('\n{}'.format(clusters_df.head()))

        # drop id, cluster not used here
        clusters_df.drop('id', axis='columns', inplace=True)
        clusters_df.drop('cluster', axis='columns', inplace=True)

        # find all unique lat/lon values
        lat = clusters_df['lat'].to_numpy()
        lon = clusters_df['lon'].to_numpy()
        unique_lat = np.unique(lat)
        unique_lon = np.unique(lon)

        # find limits and stride for grid
        (lat_min, lat_max, lat_stride) = (np.min(lat), np.max(lat), unique_lat[1] - unique_lat[0])
        (lon_min, lon_max, lon_stride) = (np.min(lon), np.max(lon), unique_lon[1] - unique_lon[0])

        # create a (lat_l*lon_l, 2) array where each row is a possible (lat, lon) coordinate
        lat_grid = np.arange(lat_min, lat_max + lat_stride, lat_stride, dtype=np.float64)
        lon_grid = np.arange(lon_min, lon_max + lon_stride, lon_stride, dtype=np.float64)

        ndarray_by_coord_2d = clusters_df.values

        np_dataset_3d = np.empty((lat_grid.shape[0], lon_grid.shape[0], 73), dtype=np.float64)
        np_dataset_3d[:] = np.NaN

        series_count = 0
        for row_index in range(ndarray_by_coord_2d.shape[0]):
            row = ndarray_by_coord_2d[row_index, :]
            lat_value, lon_value, pentads = row[0], row[1], row[2:]

            where_is_lat_value = np.where(lat_grid == lat_value)[0]
            where_is_lon_value = np.where(lon_grid == lon_value)[0]

            assert where_is_lat_value.size > 0, "This latitude is not in the grid! {}".format(lat_value)
            assert where_is_lon_value.size > 0, "This longitude is not in the grid! {}".format(lon_value)

            np_dataset_3d[where_is_lat_value[0], where_is_lon_value[0]] = pentads
            series_count = series_count + 1

        self.logger.info('Read CHIRPS 3D: {}'.format(np_dataset_3d.shape))
        self.logger.debug('Added {} elements'.format(series_count))
        return np_dataset_3d

    def cache_filename(self, temporal_md):
        '''
        The path of a file containing a temporal slice as a numpy array.
        '''
        return CACHE_FORMAT_3D.format(temporal_md)

    def retrieve(self, temporal_md):
        '''
        Returns a temporal slice of this dataset, described by the supplied metadata.
        Tries to read the data from the cache, and if it is not available it save the
        dataset slice into the cache for later use.

        TODO modify FileDataset base class implementation?
        '''

        # only support the original temporal metadata
        if temporal_md != self.dataset_temporal_md:
            log_msg = 'Cannot support {} as temporal metadata for CHIRPS, must be: {}'
            raise ValueError(log_msg.format(temporal_md, self.dataset_temporal_md))

        # if this succeeds, we are done
        dataset_slice, cache_filename = self.try_cache(temporal_md)
        if dataset_slice is not None:
            self.logger.info('Using cached dataset slice: {}'.format(cache_filename))
            return dataset_slice

        # implementation to get the dataset
        dataset_slice = self.read_from_csv()

        # save this dataset slice to cache for performance
        self.save_to_cache(dataset_slice, temporal_md)

        log_msg = 'Read dataset slice: {} -> {}'
        self.logger.info(log_msg.format(cache_filename, dataset_slice.shape))

        return dataset_slice


if __name__ == '__main__':

    import logging
    import time
    import sys

    if len(sys.argv) < 2:
        print('Usage: {} [2D|3D]'.format(sys.argv[0]))
        sys.exit(1)

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    t_start = time.time()

    dataset_type = sys.argv[1]
    assert dataset_type == '2D' or dataset_type == '3D'

    temporal_md = TemporalMetadata(2010, 2018, AveragePentads())

    if dataset_type == '2D':
        chirps_dataset = DatasetCHIRPS2D()
        chirps_dataset.retrieve(temporal_md)

    else:
        chirps_dataset = DatasetCHIRPS3D()
        chirps_numpy = chirps_dataset.retrieve(temporal_md)
        # print(chirps_numpy[0:5, 0, :])

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: {} seconds'.format(elapsed))
