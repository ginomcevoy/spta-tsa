from spta.arima import ArimaSuiteParams


def predefined_arima_suites():

    # add ARIMA suite experiments here
    # the syntax is [p_values, q_values, d_values]
    arima_suites = {
        'arima_simple': [(2,), (0,), (2,)],
        'arima_1_1_1': [(1,), (2,), (3,)],
        'arima_two': [(2,), (0,), range(1, 3)],
        'arima_sweep': [[0, 1, 2, 4, 6, 8, 10], range(0, 3), range(0, 3)]
    }

    return arima_suites


def arima_suite_by_name(suite_name):
    # get the suite parameters
    (p_values, d_values, q_values) = predefined_arima_suites()[suite_name]

    # return the instance with a generator for the suite
    # usage:
    # arima_suite = arima_suite_by_name(suite_name)
    # for arima_params in arima_suite.arima_params_gen():
    #     (use ArimaParams here)
    return ArimaSuiteParams(p_values, d_values, q_values)
