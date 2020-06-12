from spta.arima import ArimaSuiteParams, AutoArimaParams


def predefined_arima_suites():

    # add ARIMA suite experiments here
    # the syntax is [p_values, q_values, d_values]
    arima_suites = {
        'arima_simple': [(2,), (1,), (1,)],
        'arima_1_1_1': [(1,), (1,), (1,)],
        'arima_two': [(2,), (0,), range(1, 3)],
        'arima_1_2_1': [(1,), (2,), (1,)],
        'arima_2_2_2': [(2,), (2,), (2,)],
        'arima_sweep': [[0, 1, 2, 3], range(0, 2), range(0, 3)]
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


def predefined_auto_arima():

    # add auto_arima experiments here
    # the syntax is AutoArimaParams(start_p, start_q, max_p, max_q, d, stepwise)
    # if d=None, auto_arima finds 'best' d

    auto_arima_ids = {
        'simple': AutoArimaParams(1, 1, 3, 3, None, True),
    }

    return auto_arima_ids
