import logging
import numpy as np
import matplotlib.pyplot as plt

import oapackage

RESULTS_INPUT = 'raw/performance.npy'
COST_STEPS = 10

# based on https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html


def find_best_models(results_data, show=True):
    '''
    Given computational cost and prediction error, find the 'best' models,
    using Pareto optimality
    '''
    log = logging.getLogger()
    log.debug(results_data)
    (samples, cols) = results_data.shape

    # last two columns are time and prediction error, respectively
    perf_points = results_data[:, -2:]

    # plt.plot(perf_points[:, 0], perf_points[:, 1], '.b', markersize=16,
    #              label='Non Pareto-optimal')
    # _ = plt.title('The input data', fontsize=15)
    # plt.xlabel('Objective 1', fontsize=16)
    # plt.ylabel('Objective 2', fontsize=16)
    # plt.show()

    pareto = oapackage.ParetoDoubleLong()

    for ii in range(0, samples):
        # invert to 'minimize' instead of 'maximize' objectives
        w = oapackage.doubleVector((1.0 / perf_points[ii, 0], 1.0 / perf_points[ii, 1]))
        # w = oapackage.doubleVector((perf_points[ii, 0], perf_points[ii, 1]))
        pareto.addvalue(w, ii)

    pareto.show(verbose=1)

    lst = pareto.allindices()  # the indices of the Pareto optimal designs
    log.info('Found best indices: %s' % str(lst))

    optimal_datapoints = perf_points[lst, :]

    # print(optimal_datapoints[:, 0])

    # plt.plot(perf_points[:, 0], perf_points[:, 1], '.b', markersize=16,
    #          label='Non Pareto-optimal')
    # plt.show()
    # plt.plot(optimal_datapoints[:, 0], optimal_datapoints[:, 1], '.r', markersize=16,
    #          label='Pareto optimal')
    # plt.xlabel('Objective 1', fontsize=16)
    # plt.ylabel('Objective 2', fontsize=16)
    # # plt.xticks([])
    # # plt.yticks([])
    # _ = plt.legend(loc=3, numpoints=1)
    # plt.show()

    plt.scatter(perf_points[:, 0], perf_points[:, 1], c='blue')
    plt.scatter(optimal_datapoints[:, 0], optimal_datapoints[:, 1], c='red')
    if show:
        plt.show()

    # recover the other values
    optimal_results = results_data[lst, :]
    log.info('Optimal results:\n%s' % str(optimal_results))
    return optimal_results


def scale_perf_values(perf_values):
    '''
    Scale performance values so that they are between 0 and 1,
    # but keep relative weight within each metric.
    '''
    scale_factors = np.max(perf_values, axis=0)
    scaled_values = perf_values / scale_factors
    # scale_factors = np.max(perf_values, axis=0) - np.min(perf_values, axis=0)
    # scaled_values = (perf_values - np.min(perf_values, axis=0)) / scale_factors
    return (scaled_values, scale_factors)


def sweep_cost_factors(optimal_results):
    '''
    Test for several values of alpha and beta in:
    Cost = alpha * computational + beta * error

    to simplify, scale the values so that we compute:
    Cost = gamma * computational + (1 -gamma) * error
    '''
    log = logging.getLogger()

    perf_values = optimal_results[:, -2:]

    # results_shape = optimal_results.shape
    # report = np.empty((results_shape[0], results_shape[1] + 3))

    (scaled_values, scale_factors) = scale_perf_values(perf_values)
    print((scaled_values, scale_factors))
    for step in range(0, COST_STEPS):

        # between 0 and 1
        gamma = 1.0 * step / COST_STEPS

        costs = gamma * scaled_values[:, 0] + (1 - gamma) * scaled_values[:, 1]
        alpha = gamma / scale_factors[0]
        beta = (1 - gamma) / scale_factors[1]

        msg = 'cost for model: %s is %s with (alpha, beta) = %s'
        for i in range(0, optimal_results.shape[0]):
            model = optimal_results[i, :]
            cost = costs[i]
            log.info(msg % (str(model), str(cost), (alpha, beta)))

        min_cost_index = costs == np.min(costs)
        msg = 'best model is %s with cost %s for (alpha, beta) = %s'
        log.info(msg % (str(optimal_results[min_cost_index, :]), str(cost), (alpha, beta)))


if __name__ == '__main__':
    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    results_data = np.load(RESULTS_INPUT)
    optimal_results = find_best_models(results_data, False)
    sweep_cost_factors(optimal_results)
