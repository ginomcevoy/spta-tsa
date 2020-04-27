import logging
import numpy as np
import os
import sys

from spta.region import Region
from spta.region.temporal import SpatioTemporalRegion

from spta.distance.dtw import DistanceByDTW


CMD_OPTIONS = ('save', 'distances')

# NORDESTE:     20:50, 65:95
# SP_RIO:       40:75, 50:85
# SP:           55:75, 50:70
# BRIAN?:       40:80, 45:85

SP_RJ_REGION = Region(40, 75, 50, 85)
SP_RJ_DATASET = 'raw/sp_rj_1y_1ppd.npy'
SP_RJ_DISTANCES = 'raw/distances_sp_rj_1y_1ppd.npy'

logger = logging.getLogger()


def main(cmd_option):
    '''
    Choose option
    '''
    if cmd_option == 'save':
        save_sp_rj_dataset()
    else:
        calculate_distances()


def save_sp_rj_dataset():
    '''
    Saves the SP/RJ dataset into a file
    '''
    dataset_1y_1ppd = SpatioTemporalRegion.load_1year_1ppd()
    sp_rj_region = dataset_1y_1ppd.region_subset(SP_RJ_REGION)
    sp_rj_region.save(SP_RJ_DATASET)


def calculate_distances():
    # load file
    sp_rj_dataset = np.load(SP_RJ_DATASET)
    sp_rj_region = SpatioTemporalRegion(sp_rj_dataset)

    logger.info('Calculating distances using DTW...')
    distance_measure = DistanceByDTW()
    distance_matrix = distance_measure.compute_distance_matrix(sp_rj_region)
    np.save(SP_RJ_DISTANCES, distance_matrix)


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in CMD_OPTIONS:

        _, filename = os.path.split(sys.argv[0])
        opts_str = '|'.join(CMD_OPTIONS)
        print('Usage: {} [{}]'.format(filename, opts_str))
        sys.exit(1)

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    main(sys.argv[1])
