from collections import namedtuple

ArimaPDQ = namedtuple('ArimaPDQ', 'p d q')


class AutoArimaParams(namedtuple('AutoArimaParams', 'start_p start_q max_p max_q d stepwise')):
    '''
    Hyper parameters for an auto ARIMA model
    '''
    __slots__ = ()

    def __str__(self):
        '''
        Override the string representation of AutoArimaParams
        # https://stackoverflow.com/a/7914212/3175179
        '''
        as_str = 'start_p{}-start_q{}-max_p{}-max_q{}-d{}-stepwise{}'
        return as_str.format(self.start_p, self.start_q, self.max_p, self.max_q, self.d,
                             self.stepwise)


class ArimaSuiteParams(object):
    '''
    Generates a list of ArimaPDQ instances with (p, d, q) tuples, by sweeping the values
    in the lists for p, d, q.

    E.g.
    p_values = (1, 2)
    d_values = (0, 2)
    q_values = (0, 1)

    -> (1, 0, 0), (1, 0, 1), (1, 2, 0), ... (2, 2, 1)
    '''
    def __init__(self, p_values, d_values, q_values):
        self.p_values = p_values
        self.d_values = d_values
        self.q_values = q_values

    def arima_params_gen(self):
        '''
        Generator pattern to return the next instance of ArimaPDQ
        '''
        for p in self.p_values:
            for d in self.d_values:
                for q in self.q_values:
                    yield ArimaPDQ(p, d, q)
