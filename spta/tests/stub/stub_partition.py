import numpy as np


def crisp_membership_stub():
    '''
    A 2-d array indicating membership to a cluster, from a crisp clustering algorithm with k=3.
    Assumes a 4x5 region.
    '''
    partition_np = np.empty((4, 5))

    partition_np[0, 0] = 0
    partition_np[0, 1] = 1
    partition_np[0, 2] = 1
    partition_np[0, 3] = 0
    partition_np[0, 4] = 0
    partition_np[1, 0] = 1
    partition_np[1, 1] = 1
    partition_np[1, 2] = 2
    partition_np[1, 3] = 2
    partition_np[1, 4] = 2
    partition_np[2, 0] = 0
    partition_np[2, 1] = 2
    partition_np[2, 2] = 1
    partition_np[2, 3] = 1
    partition_np[2, 4] = 2
    partition_np[3, 0] = 0
    partition_np[3, 1] = 0
    partition_np[3, 2] = 1
    partition_np[3, 3] = 1
    partition_np[3, 4] = 2

    return partition_np


def fuzzy_membership_stub():
    '''
    A 3-d array indicating membership to a cluster, from a fuzzy clustering algorithm with k=2.
    It is a subset of the output of k-medoids fuzzy for the nordeste_small_1y_1ppd dataset,
    with k=2, seed=0. The subset has 20 points and has the medoids at indices 0 and 19.
    '''
    return np.array([[0.00000000, 1.00000000],
                     [0.30769328, 0.69230672],
                     [0.40253355, 0.59746645],
                     [0.43226533, 0.56773467],
                     [0.52156374, 0.47843626],
                     [0.51702938, 0.48297062],
                     [0.53440908, 0.46559092],
                     [0.58319551, 0.41680449],
                     [0.61056343, 0.38943657],
                     [0.48481703, 0.51518297],
                     [0.40998361, 0.59001639],
                     [0.40611098, 0.59388902],
                     [0.43777396, 0.56222604],
                     [0.45104678, 0.54895322],
                     [0.52606050, 0.47393950],
                     [0.51337058, 0.48662942],
                     [0.54337398, 0.45662602],
                     [0.62787528, 0.37212472],
                     [1.00000000, 0.00000000],
                     [0.62159693, 0.37840307]])


def mask_fuzzy_np_stub():
    '''
    The numpy representation of a fuzzy partition, built manually from fuzzy_membership_stub data,
    assuming a 4x5 Region.
    '''
    partition_np = np.empty((2, 4, 5))

    partition_np[:, 0, 0] = [0.00000000, 1.00000000]
    partition_np[:, 0, 1] = [0.30769328, 0.69230672]
    partition_np[:, 0, 2] = [0.40253355, 0.59746645]
    partition_np[:, 0, 3] = [0.43226533, 0.56773467]
    partition_np[:, 0, 4] = [0.52156374, 0.47843626]
    partition_np[:, 1, 0] = [0.51702938, 0.48297062]
    partition_np[:, 1, 1] = [0.53440908, 0.46559092]
    partition_np[:, 1, 2] = [0.58319551, 0.41680449]
    partition_np[:, 1, 3] = [0.61056343, 0.38943657]
    partition_np[:, 1, 4] = [0.48481703, 0.51518297]
    partition_np[:, 2, 0] = [0.40998361, 0.59001639]
    partition_np[:, 2, 1] = [0.40611098, 0.59388902]
    partition_np[:, 2, 2] = [0.43777396, 0.56222604]
    partition_np[:, 2, 3] = [0.45104678, 0.54895322]
    partition_np[:, 2, 4] = [0.52606050, 0.47393950]
    partition_np[:, 3, 0] = [0.51337058, 0.48662942]
    partition_np[:, 3, 1] = [0.54337398, 0.45662602]
    partition_np[:, 3, 2] = [0.62787528, 0.37212472]
    partition_np[:, 3, 3] = [1.00000000, 0.00000000]
    partition_np[:, 3, 4] = [0.62159693, 0.37840307]

    return partition_np
