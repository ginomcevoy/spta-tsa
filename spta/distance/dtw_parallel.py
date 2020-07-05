import numpy as np

from spta.util import parallel as parallel_util
from spta.util import log as log_util

from .dtw import DistanceByDTW


# MAYBE improve performance by assigning rows to processes, instead of single points?

class DistanceByDTWParallel(DistanceByDTW):

    def __init__(self, num_proc):
        super(DistanceByDTWParallel, self).__init__()
        self.num_proc = num_proc

    def compute_distance_matrix_sptr_old(self, spatio_temporal_region):

        _, x_len, y_len = spatio_temporal_region.shape
        sptr_2d = spatio_temporal_region.as_2d
        shape_2d = (x_len * y_len, x_len * y_len)

        def dtw_i_j(i, j, sptr_2d):
            if i <= j:
                # since distance_matrix is triangular and the diag is zero,
                # we don't need all values, store 0
                return 0
            else:
                # here we calculate DTW
                return super(DistanceByDTWParallel, self).measure(sptr_2d[i], sptr_2d[j])

        parallel_queue = parallel_util.Parallel2DQueue(self.num_proc)
        distance_matrix = parallel_queue.execute_func_on_2D(dtw_i_j, sptr_2d, shape_2d)

        # complete matrix to reflect both sides
        distance_matrix = distance_matrix + np.transpose(distance_matrix)
        return distance_matrix

    def compute_distance_matrix_sptr(self, spatio_temporal_region):
        '''
        Parallel computation of the distance matrix with DTW
        '''

        # _, x_len, y_len = spatio_temporal_region.shape
        # sptr_2d = spatio_temporal_region.as_2d
        # shape_2d = (x_len * y_len, x_len * y_len)
        inter_points_op = parallel_util.InterPointsOperation(self.num_proc, spatio_temporal_region)
        distance_matrix = inter_points_op.operate(dtw_i_j)

        # complete matrix to reflect both sides
        distance_matrix = distance_matrix + np.transpose(distance_matrix)
        return distance_matrix


def dtw_i_j(spt_region, k1, k2):

    if k1 <= k2:
        # since distance_matrix is triangular and the diag is zero,
        # we don't need all values, store 0
        return 0
    else:

        # print for first time i gets to do work
        if k1 == k2 + 1:
            print('k1={}'.format(k1), flush=True)

        # here we calculate DTW
        sptr_2d = spt_region.as_2d
        return DistanceByDTW().measure(sptr_2d[k1], sptr_2d[k2])


if __name__ == '__main__':

    import numpy as np
    import logging

    from spta.region import Region
    from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalRegionMetadata

    logger = log_util.setup_log('DEBUG')

    # load dataset from metadata

    nordeste_small_md = SpatioTemporalRegionMetadata('nordeste_small', Region(43, 50, 85, 95),
                                                     series_len=365, ppd=1, last=True)
    nordeste_small_region = SpatioTemporalRegion.from_metadata(nordeste_small_md)

    logger.info('Calculating distances using DTW...')
    distance_measure_parallel = DistanceByDTWParallel(4)
    distance_matrix = distance_measure_parallel.compute_distance_matrix(nordeste_small_region)
    print(distance_matrix)

    # compare with known result
    # it is not really equal because of triangular approach...
    dtw = DistanceByDTW()
    distance_saved = dtw.load_distance_matrix_md(nordeste_small_md)
    print(distance_saved)

    print(distance_saved - distance_matrix)
