from datetime import datetime
import numpy as np
from math import sqrt


def divisors(n):
    '''
    Compute the divisors of a number, returns them in order.
    '''
    divs = {1, n}
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            divs.update((i, n // i))
    return sorted(divs)


def find_two_balanced_divisors(n):
    '''
    Given n, compute x, y integers such that x*y = n, with the property that they are the
    the most "balanced", i.e. closest to each other.

    Uses divisors and iterates them. For each divisor d, finds the quotient n/d. Stops when the
    quotient is not bigger than d.
    '''
    divs = divisors(n)

    # handle prime case
    if len(divs) == 2:
        return divs

    # iterate the divisors, find the moment where the quotient is not bigger
    for div in divs:
        quotient = n / div
        if quotient <= div:
            break

    return [int(quotient), int(div)]


def two_balanced_divisors_order_x_y(n, x_len, y_len):
    '''
    Similar to find_two_balanced_divisors, but orders the divisors in the same order as the
    sizes of (x_len, y_len).
    '''
    (div_1, div_2) = find_two_balanced_divisors(n)

    if x_len < y_len:
        div_x, div_y = div_1, div_2
    else:
        div_x, div_y = div_2, div_1

    return (div_x, div_y)


def random_integers_with_blacklist(n, min_value, max_value, blacklist=[]):
    '''
    Given a max value and a blacklist, find n different random integers (uniformly distributed),
    located between [min_value, max_value] (inclusive) but that do not match any integers provided
    in the blacklist. Outputs a numpy array of integers size n. Will fail if there are not enough
    points available for sampling!

    n
        number of random integers to produce

    min_value, max value
        integers that determine the [min_value, max_value] integer

    blacklist
        array with integers that are blacklisted from search
    '''
    all_values = range(min_value, max_value + 1)
    allowed = set(all_values).difference(set(blacklist))
    allowed = list(allowed)
    return np.random.choice(allowed, size=n, replace=False)


def days_in_year_interval(year_start, year_end):
    """
    Calculates the number of days contained in a year interval. If both years are the same,
    then it returns the number days in that year (not zero).

    This takes leap years into account!

    E.g. 1979, 1980 -> returns 365 + 366 = 731
    """
    # use Python calendar to take leap years into account
    date_year_start = datetime(year_start, 1, 1)
    date_year_end = datetime(year_end + 1, 1, 1)   # exact end of the end year

    # calculate the number of days between the years
    day_offset = (date_year_end - date_year_start).days
    return day_offset


def years_to_series_interval(year_start, year_end, first_year_in_sample, samples_per_day):
    '''
    Given a time series with samples_per_day (e.g. 1460 samples per non-leap year,
    given 4 samples per day), and a start and end year, calculate the corresponding interval.

    This takes leap years into account!

    Returns a tuple (series_start, series_end).

    year_start
        The first sample in the series interval must correspond at the beginning of this year.

    year_end
        The last sample in the series interval must correspond to the end of this year.
        This means that year_start = 2014, year_end = 2014 will contain 1 year of data.

    samples_per_day
        number of samples per day in the dataset, needs to be an integer.

    First example:
        year_start = 1979
        year_end = 1979
        first_year_in_sample = 1979
        samples_per_day = 4

        Then we have (0, 1460) where 1460 = (365 * 4)

    Second example:
        year_start = 1980
        year_end = 1980
        first_year_in_sample = 1979
        samples_per_day = 4

        Then we have (1460, 2924) where 2924 = (1460 + 366*4, leap year)

    Third example:
        year_start = 2014
        year_end = 2015
        first_year_in_sample = 1979
        samples_per_day = 4

        Then we have (51136, 54056), 2920 samples, there are 9 leap years in between 1979 and 2014.

    '''
    # only works for integers
    assert isinstance(samples_per_day, int)

    # find where year before year_start ends
    day_offset_for_year_start = days_in_year_interval(first_year_in_sample, year_start - 1)

    # find where year after year_end begins
    day_offset_for_year_end = days_in_year_interval(first_year_in_sample, year_end)

    # apply number of samples
    return (day_offset_for_year_start * samples_per_day, day_offset_for_year_end * samples_per_day)
