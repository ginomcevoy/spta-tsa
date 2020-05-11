from spta.arima import ArimaParams


def predefined_arima_experiments():

    # add ARIMA experiments here
    arima_experiments = {
        'arima_simple': [ArimaParams(2, 0, 2)]
    }

    return arima_experiments


def arima_experiment_by_name(experiment_name):
    return predefined_arima_experiments()[experiment_name]
