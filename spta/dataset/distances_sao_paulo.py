import logging
import numpy as np

from spta.region.temporal import SpatioTemporalRegion
from spta.region.distance import DistanceByDTW

DISTANCES_SAO_PAULO = 'raw/distances_sao_paulo_1y_1ppd.npy'


def main():

    # load the dataset for last year of sao paulo, then calculate the distance matrix.
    spt_region = SpatioTemporalRegion.load_sao_paulo()

    distance_measure = DistanceByDTW()

    logger = logging.getLogger()
    logger.info('Calculating distances using DTW...')
    distance_matrix = distance_measure.compute_distance_matrix(spt_region)

    logger.info('Saving distance matrix to: {}'.format(DISTANCES_SAO_PAULO))
    np.save(DISTANCES_SAO_PAULO, distance_matrix)

    # load to test
    logger.info('Loading distance matrix again: {}'.format(DISTANCES_SAO_PAULO))
    recovered = np.load(DISTANCES_SAO_PAULO)
    logger.info('Recovered distance matrix: {}'.format(recovered.shape))


if __name__ == '__main__':

    log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    main()
