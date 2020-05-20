

def arima_clustering_experiments():

    # add ARIMA clustering experiments here
    # arima_clustering_id = [<arima_experiment_id>, 'Kmedoids', distance_id, k, seed]

    return {
        'arima_simple_2_0': ['arima_simple', 'Kmedoids', 'DistanceByDTW', 2, 0],
        'arima_two_2_0': ['arima_two', 'Kmedoids', 'DistanceByDTW', 2, 0],
        'arima_whole_brazil': ['arima_sweep', 'Kmedoids', 'DistanceByDTW', 8, 0],
        'arima_whole_brazil_fails': ['arima_2_2_2', 'Kmedoids', 'DistanceByDTW', 8, 0]
    }


def arima_clustering_experiment_by_name(experiment_name):
    return arima_clustering_experiments()[experiment_name]
