'''
Create some synthetic data to test k-medoids
Based on https://github.com/fpetitjean/DBA/blob/master/DBA.py
(Copyright (C) 2018 Francois Petitjean)
'''

import numpy as np
import matplotlib.pyplot as plt


def delayed_function_with_noise(main_profile_func, series_len):
    '''
    Given a function, computes a function with a randomized delay, with 2% of white noise added
    on top.
    '''

    # up to 15% of the series length in delays, maximum is 30
    max_delay_indices = 30
    min_delay_ratio = 0.15
    padding_length = int(min(max_delay_indices, min_delay_ratio * series_len))

    # number of function points
    indices = range(0, series_len - padding_length)

    # evaluate the function in the available points
    main_profile_gen = np.array([main_profile_func(j, indices) for j in indices])
    # main_profile_gen = np.array([np.sin(2 * np.pi * j / len(indices)) for j in indices])

    # to add noise
    def randomizer(x):
        return np.random.normal(x, 0.02)
    randomizer_fun = np.vectorize(randomizer)

    # for i in range(0, n_series):

    # the delay is padded with noise
    n_pad_left = np.random.randint(0, padding_length)

    # adding zero at the start or at the end to shif the profile
    series = np.pad(main_profile_gen, (n_pad_left, padding_length - n_pad_left),
                    mode='constant', constant_values=0)

    # randomize a bit
    series = randomizer_fun(series)

    return series


def sine_function(j, indices):
    return np.sin(2 * np.pi * j / len(indices))


def square_function(j, indices):
    return np.sign(sine_function(j, indices))


def gaussian_function(j, indices):
    from scipy.stats import norm

    mu = 0
    sigma = 1
    x = 3 * sigma * (2 * j / len(indices) - 1)

    return norm.pdf(x, mu, sigma)


if __name__ == '__main__':

    # create two series for each function
    series_len = 200
    sine_series1 = delayed_function_with_noise(sine_function, series_len)
    sine_series2 = delayed_function_with_noise(sine_function, series_len)
    square_series1 = delayed_function_with_noise(square_function, series_len)
    square_series2 = delayed_function_with_noise(square_function, series_len)

    # plot them all
    plt.figure()
    plt.plot(range(0, series_len), sine_series1)
    plt.plot(range(0, series_len), sine_series2)
    plt.plot(range(0, series_len), square_series1)
    plt.plot(range(0, series_len), square_series2)
    plt.show()
