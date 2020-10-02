import numpy as np
import pandas as pd


def apply_sliding_window_to_labeled_series(df, window_size):
    '''
    Given a pandas dataframe where each tuple has a label a time series, returns a new pandas
    dataframe where each tuple has a label and one element of a sliding window passing applied
    to the time series. Also returns a new column for the window offset.

    Example of a single tuple
    --------------------------
    Input: this dataframe and window_size=3

      label    s0    s1    s2    s3    s4    s5    s6
    0     a   0.0   1.0   2.0   3.0   4.0   5.0   6.0

    Expected output:
      label           window  offset
    0     a  [0.0, 1.0, 2.0]       0
    1     a  [1.0, 2.0, 3.0]       1
    2     a  [2.0, 3.0, 4.0]       2
    3     a  [3.0, 4.0, 5.0]       3
    4     a  [4.0, 5.0, 6.0]       4

    Example of two tuples
    ---------------------
    Input: this dataframe and window_size=3

      label     s0     s1     s2     s3     s4     s5     s6
    0     a    0.0    1.0    2.0    3.0    4.0    5.0    6.0
    1     b   10.0   11.0   12.0   13.0   14.0   15.0   16.0

    Expected output:
      label              window  offset
    0     a     [0.0, 1.0, 2.0]       0
    1     a     [1.0, 2.0, 3.0]       1
    2     a     [2.0, 3.0, 4.0]       2
    3     a     [3.0, 4.0, 5.0]       3
    4     a     [4.0, 5.0, 6.0]       4
    5     b  [10.0, 11.0, 12.0]       0
    6     b  [11.0, 12.0, 13.0]       1
    7     b  [12.0, 13.0, 14.0]       2
    8     b  [13.0, 14.0, 15.0]       3
    9     b  [14.0, 15.0, 16.0]       4

    NOTE: it is possible that a = b
    '''
    tuples = df.shape[0]
    series_len = df.shape[1] - 1

    # work with each tuple
    partial_results = [
        extract_windows_from_one_tuple(df.iloc[[t]], window_size, series_len)
        for t in range(0, tuples)
    ]

    result = pd.concat(partial_results)
    result = result.reset_index(drop=True)
    return result

def extract_windows_from_one_tuple(df_one, window_size, series_len):

    # melt the series: this returns num(labels) * num(windows) tuples
    value_vars = ['s' + str(i) for i in range(0, series_len)]
    df_melted = df_one.melt(id_vars=['label'], value_vars=value_vars, value_name='Temp')
    df_melted = df_melted.reset_index(drop=True)

    # apply sliding window
    list_of_values = []
    list_of_values = []
    df_melted.Temp.rolling(window_size).apply(lambda x: list_of_values.append(x) or 0, raw=True)
    window_elems = pd.Series(list_of_values).values
    window_elems = np.concatenate((np.zeros((window_size - 1,)), window_elems))
    df_melted['window'] = window_elems

    # remove window_size - 1 first rows which don't have a window
    df_melted = df_melted.iloc[window_size - 1:]
    df_melted = df_melted.reset_index(drop=True)

    # recover the offset: always start at zero, as many as windows can be fit
    num_windows = series_len - window_size + 1
    df_melted['offset'] = np.arange(0, num_windows)

    # keep only the columns we want
    df_partial_result = df_melted.loc[:, ['label', 'window', 'offset']]
    return df_partial_result
