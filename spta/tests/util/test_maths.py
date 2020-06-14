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
        self.assertEqual(divs, [1, 7])

    def test_find_two_balanced_divisors_4(self):
        # given
        n = 4

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 2x2
        self.assertEqual(divs, [2, 2])

    def test_find_two_balanced_divisors_30(self):
        # given
        n = 30

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 5x6
        self.assertEqual(divs, [5, 6])

    def test_find_two_balanced_divisors_100(self):
        # given
        n = 100

        # when
        divs = maths_util.find_two_balanced_divisors(n)

        # then 10x10
        self.assertEqual(divs, [10, 10])


class TestRandomIntegersWithBlacklist(unittest.TestCase):
    '''
    Unit tests for maths.random_integers_with_blacklist function.
    '''

    def test_find_only_one(self):
        # given an interval with only one permissible value
        n = 1
        min_value = 0
        max_value = 0
        blacklist = []

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then only value is returned
        self.assertEqual(set(result), set([0]))
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def test_find_only_two(self):
        # given an interval with only two permissible values
        n = 2
        min_value = 0
        max_value = 1
        blacklist = []

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then only value is returned
        self.assertEqual(set(result), set([0, 1]))
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def test_find_two_out_of_three(self):
        # given an interval with three permissible values, but 2 are requested
        n = 2
        min_value = 0
        max_value = 2
        blacklist = []

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then two values are returned
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def test_find_three_out_of_ten(self):
        # given an interval with 10 permissible values, but 3 are requested
        n = 3
        min_value = 1
        max_value = 10
        blacklist = []

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then two values are returned
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def test_find_only_three_allowed(self):
        # given an interval with 6 permissible values, 3 are requested and 3 are blacklisted
        n = 3
        min_value = 0
        max_value = 5
        blacklist = [1, 3, 4]

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then only three remaining are chosen
        self.assertEqual(set(result), set([0, 2, 5]))
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def test_find_five_out_of_twenty_with_blacklist(self):
        # given an interval with 20 permissible values, 5 are requested and 8 are blacklisted
        n = 5
        min_value = 1
        max_value = 20
        blacklist = [1, 3, 4, 7, 8, 11, 14, 16]

        # when
        result = maths_util.random_integers_with_blacklist(n, min_value, max_value, blacklist)

        # then five permissible integers are chosen
        self.evaluate_random_array_of_integers(result, n, min_value, max_value, blacklist)

    def evaluate_random_array_of_integers(self, result, n, min_value, max_value, blacklist):
        '''
        Evaluates that a random array resulting from random_integers_with_blacklist is plausible.
        The following rules are checked:

        1. the size matches n
        2. all values are different
        3. numbers are integers within specified interval
        4. None of the numbers are in blacklist
        '''
        # 1. the size matches n
        self.assertEqual(len(result), n)

        # build sets
        result_set = set(result)
        range_set = set(range(min_value, max_value + 1))  # range of integers, allow max_value
        blacklist_set = set(blacklist)

        # 2. all values are different
        self.assertEqual(len(result_set), n)

        # 3. numbers are integers within specified interval
        # {numbers} - {interval numbers} = {} (empty set)
        self.assertEqual(result_set.difference(range_set), set())

        # 4. None of the numbers are in blacklist
        self.assertEqual(result_set.intersection(blacklist_set), set())
