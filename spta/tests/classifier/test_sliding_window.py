'''
Unit tests for spta.classifer.sliding_window module.
'''

import pandas as pd
import unittest

from spta.classifier import sliding_window


class TetsApplySlidingWindowToLabeledSeries(unittest.TestCase):
    '''
    Unit tests for sliding_window.apply_sliding_window_to_labeled_series() method.
    '''

    def setUp(self):
        self.df_one = pd.DataFrame(
            {'label': ['a', ],
             's0': [0., ], 's1': [1., ], 's2': [2., ], 's3': [3., ], 's4': [4., ],
             's5': [5., ], 's6': [6., ]
             })

        self.expected_one_ws3 = pd.DataFrame(
            {'label': ['a', 'a', 'a', 'a', 'a'],
             'window': [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]],
             'offset': [0, 1, 2, 3, 4]
             })

        self.expected_one_ws4 = pd.DataFrame(
            {'label': ['a', 'a', 'a', 'a'],
             'window': [[0., 1., 2., 3.], [1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6]],
             'offset': [0, 1, 2, 3]
             })

        self.df_two = pd.DataFrame(
            {'label': ['a', 'b'],
             's0': [0., 10.], 's1': [1., 11], 's2': [2., 12], 's3': [3., 13], 's4': [4., 14],
             's5': [5., 15], 's6': [6., 16]})

        self.expected_two_ws3 = pd.DataFrame(
            {'label': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
             'window': [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.],
                        [10., 11., 12.], [11., 12., 13.], [12., 13., 14.], [13., 14., 15.], [14., 15., 16.]],
             'offset': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
             })

        self.df_complex = pd.DataFrame(
            {'label': ['a', 'b', 'a'],
             's0': [0., 10., 20.], 's1': [1., 11., 21], 's2': [2., 12., 22.], 's3': [3., 13., 23.], 's4': [4., 14., 24.],
             's5': [5., 15., 25.], 's6': [6., 16., 26.]})

        self.df_complex_ws3 = pd.DataFrame(
            {'label': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'a'],
             'window': [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.],
                        [10., 11., 12.], [11., 12., 13.], [12., 13., 14.], [13., 14., 15.], [14., 15., 16.],
                        [20., 21., 22.], [21., 22., 23.], [22., 23., 24.], [23., 24., 25.], [24., 25., 26.]],
             'offset': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
             })

    def test_single_label_ws3(self):
        # given
        df = self.df_one
        window_size = 3

        # when
        result = sliding_window.apply_sliding_window_to_labeled_series(df, window_size=window_size)

        # then
        pd.testing.assert_frame_equal(result, self.expected_one_ws3)

    def test_single_label_ws4(self):
        # given
        df = self.df_one
        window_size = 4

        # when
        result = sliding_window.apply_sliding_window_to_labeled_series(df, window_size=window_size)

        # then
        pd.testing.assert_frame_equal(result, self.expected_one_ws4)

    def test_two_labels_ws3(self):
        # given
        df = self.df_two
        window_size = 3

        # when
        result = sliding_window.apply_sliding_window_to_labeled_series(df, window_size=window_size)
        print(result)

        # then
        pd.testing.assert_frame_equal(result, self.expected_two_ws3)

    def test_complex_ws3(self):
        # given
        df = self.df_complex
        window_size = 3

        # when
        result = sliding_window.apply_sliding_window_to_labeled_series(df, window_size=window_size)
        print(result)

        # then
        pd.testing.assert_frame_equal(result, self.df_complex_ws3)
