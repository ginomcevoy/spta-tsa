from spta.clustering.suite import ClusteringSuite


def kmedoids_quick_stub():
    '''
    Stub clustering suite for kmedoids-quick
    '''
    # given kmedoids metadata
    identifier = 'quick'
    metadata_name = 'kmedoids'
    ks = range(2, 4)  # k=2, k=3
    random_seeds = range(0, 2)  # 0, 1
    return ClusteringSuite(identifier, metadata_name, k=ks, random_seed=random_seeds)
