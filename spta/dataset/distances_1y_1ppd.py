import logging
import numpy as np
import time

from spta.region.temporal import SpatioTemporalRegion
from spta.region.distance import DistanceByDTW

DISTANCES_1y_1ppd = 'raw/distances_1y_1ppd.npy'


def main():
    '''
    Load the dataset for last year of Brazil, then calculate the distance matrix and save it.
    '''
    spt_region = SpatioTemporalRegion.load_1year_1ppd()

    distance_measure = DistanceByDTW()

    logger = logging.getLogger()
    logger.info('Calculating distances using DTW...')
    distance_matrix = distance_measure.compute_distance_matrix(spt_region)

    logger.info('Saving distance matrix to: {}'.format(DISTANCES_1y_1ppd))
    np.save(DISTANCES_1y_1ppd, distance_matrix)

    # load to test
    logger.info('Loading distance matrix again: {}'.format(DISTANCES_1y_1ppd))
    recovered = np.load(DISTANCES_1y_1ppd)
    logger.info('Recovered distance matrix: {}'.format(recovered.shape))


if __name__ == '__main__':

    t_start = time.time()

    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    main()

    t_stop = time.time()
    elapsed = t_stop - t_start
    logger = logging.getLogger()
    logger.info('Elapsed: {}s'.format(elapsed))
