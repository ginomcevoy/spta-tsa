from collections import namedtuple


def arima_clustering_experiments():

    # add ARIMA clustering experiments here
    # arima_clustering_id = [<arima_experiment_id>, 'Kmedoids', distance_id, k, seed]

    return {
        'arima_simple_2_0': ['arima_simple', 'Kmedoids', 'DistanceByDTW', 2, 0],
        'arima_two_2_0': ['arima_two', 'Kmedoids', 'DistanceByDTW', 2, 0],
        'arima_whole_brazil_fuzzy_0.05': ['arima_sweep', 'KmedoidsFuzzy', 'DistanceByDTW', 8, 0,
                                          0.05],
        'arima_whole_brazil_fails': ['arima_1_2_1', 'Kmedoids', 'DistanceByDTW', 8, 0],
        'arima_whole_brazil_8': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 8, 0],
        'arima_whole_brazil_10': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 10, 0],
        'arima_whole_brazil_12': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 12, 0],
        'arima_whole_brazil_14': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 14, 0],
        'arima_whole_brazil_16': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 16, 0],
        'arima_whole_brazil_18': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 18, 0],
        'arima_whole_brazil_20': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 20, 0],
        'arima_whole_brazil_22': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 22, 0],
        'arima_whole_brazil_24': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 24, 0],
        'arima_whole_brazil_26': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 26, 0]
    }


def arima_clustering_experiment_by_name(experiment_name):
    return arima_clustering_experiments()[experiment_name]


AutoArimaCluster = namedtuple('AutoArimaCluster', 'auto_arima_id clustering kmedoids_id distance')


def auto_arima_clustering_experiments():

    # add auto_arima experiments here
    # auto_arima_clustering_id = AutoArimaCluster(<auto_arima_id>, 'Kmedoids', <kmedoids_suite_id>,
    #                                             'DistanceByDTW')

    auto_arima_clustering_id = {
        'simple_quick': AutoArimaCluster('simple', 'Kmedoids', 'quick', 'DistanceByDTW'),
        'simple_even': AutoArimaCluster('simple', 'Kmedoids', 'even', 'DistanceByDTW'),
    }

    return auto_arima_clustering_id
