import unittest

from spta.arima import ArimaPDQ, ArimaSuiteParams


class TestArimaSuiteParams(unittest.TestCase):

    def test_pdq_lists(self):

        # given a suite
        asp = ArimaSuiteParams(p_values=[0, 1, 2, 4, 6, 8, 10], d_values=range(0, 3),
                               q_values=range(0, 3))

        # when iterating through suite
        experiments = []
        for arima_params in asp.arima_params_gen():
            experiments.append(arima_params)

        # then all instances are created
        self.assertEqual(len(experiments), 63)
        self.assertEqual(experiments[0], ArimaPDQ(0, 0, 0))
        self.assertEqual(experiments[62], ArimaPDQ(10, 2, 2))
