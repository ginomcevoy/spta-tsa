
from .base import FileDataset
from .metadata import TemporalMetadata, AveragePentads


CACHE_FORMAT = 'raw/chirps_{!r}.npy'


class DatasetCHIRPS(FileDataset):
    '''CHIRPS dataset, TODO description'''

    def __init__(self):
        '''Initialize the parent FileDataset with the metadata of this dataset.'''
        time_to_series = AveragePentads()
        dataset_temporal_md = TemporalMetadata(2010, 2018, time_to_series)
        super(DatasetCHIRPS, self).__init__(dataset_temporal_md)

    def cache_filename(self, temporal_md):
        '''
        The path of a file containing a temporal slice as a numpy array.
        '''
        return CACHE_FORMAT.format(temporal_md)

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


if __name__ == '__main__':

    import logging
    import time
    import sys

    if len(sys.argv) < 3:
        print('Usage: {} <year_start> <year_end>'.format(sys.argv[0]))
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

    chirps_dataset = DatasetCHIRPS()
    temporal_md = TemporalMetadata(year_start, year_end, AveragePentads())
    chirps_dataset.retrieve(temporal_md)

    t_end = time.time()
    elapsed = t_end - t_start
    print('Elapsed: {} seconds'.format(elapsed))
