'''
Unit tests for spta.util.maths module.
'''

import unittest

from spta.util import maths as maths_util


class TestFindTwoBalancedDivisors(unittest.TestCase):
    '''
    Unit tests for maths.find_two_balanced_divisors function.
    '''

    def test_find_two_balanced_divisors_prime(self):
        # given
        n = 7

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then only options
        self.assertEquals(divs, [1, 7])

    def test_find_two_balanced_divisors_4(self):
        # given
        n = 4

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 2x2
        self.assertEquals(divs, [2, 2])

    def test_find_two_balanced_divisors_30(self):
        # given
        n = 30

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 5x6
        self.assertEquals(divs, [5, 6])

    def test_find_two_balanced_divisors_100(self):
        # given
        n = 100

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 10x10
        self.assertEquals(divs, [10, 10])
