import h5py
import os

from .base import FileDataset
from .metadata import TemporalMetadata, SamplesPerDay


RAW_DATASET = 'raw/TEMPERATURE_1979-2015.hdf'
CACHE_FORMAT = 'raw/SouthAmerica_{!r}.npy'


class DatasetCSFR(FileDataset):
    '''Temperatures from South America from 1979 to 2019, 4 samples per day'''

    def __init__(self):
        '''Initialize the parent FileDataset with the metadata of this dataset.'''
        time_to_series = SamplesPerDay(4)
        dataset_temporal_md = TemporalMetadata(1979, 2015, time_to_series)
        super(DatasetCSFR, self).__init__(dataset_temporal_md)

    def read_raw(self, series_start, series_end):
        '''
        Returns the dataset as a numpy multi-dimensional array.
        Reads from hdf5 file, entire coordinates, using the sample interval series_start:series_end.
        '''
        dataset_str = '{}[{}:{}]'.format(RAW_DATASET, series_start, series_end)
        self.logger.debug('Reading dataset: {}'.format(dataset_str))

        if not os.path.exists(RAW_DATASET):
            msg = 'Dataset not found {}'.format(RAW_DATASET)
            raise ValueError(msg)

        with h5py.File(RAW_DATASET, 'r') as f:
            real = f['real'][...]
            self.logger.info('field "real" of dataset has shape: {}'.format(real.shape))
            real = real[series_start:series_end, :, :]

        series_len = series_end - series_start
        self.logger.info('Read dataset {} with {} samples: {}'.format(dataset_str, series_len, real.shape))
        return real

    def cache_filename(self, temporal_md):
        '''
        The path of a file containing a temporal slice as a numpy array.
        '''
        return CACHE_FORMAT.format(temporal_md)


if __name__ == '__main__':

    import logging
    import time
    import sys

    if len(sys.argv) < 3:
        print('Usage: {} <year_start> <year_end> ["spd1"]'.format(sys.argv[0]))
        sys.exit(1)

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    t_start = time.time()

    year_start = int(sys.argv[1])
    year_end = int(sys.argv[2])

    spd = 4
    if len(sys.argv) > 3:
        spd = 1

    csfr_dataset = DatasetCSFR()
    time_to_series = SamplesPerDay(spd)
    temporal_md = TemporalMetadata(year_start, year_end, time_to_series)
    csfr_dataset.retrieve(temporal_md)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: {} seconds'.format(elapsed))
