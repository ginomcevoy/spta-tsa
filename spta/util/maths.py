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
