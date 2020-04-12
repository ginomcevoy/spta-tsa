#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np      # vectors and matrices
import pandas as pd                              # tables and data manipulations


import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


with h5py.File('TEMPERATURE_1979-2015.hdf', 'r') as hdf:
    # Access to the structure named 'real'
    data = hdf['real']
    # Extract time series with 400 points for all (lat, long).
    first = data[0:4000, :, :]
    print(first[:, 0, 0])
    # plt.(first[:,0,0])

# Create a small sample with 200 spatial-points and 1460 elements in the time-series
# (corresponding to 1 year)


# Function to create ARIMA
# Reference:
# https://towardsdatascience.com/forecasting-exchange-rates-using-arima-in-python-f032f313fc56
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


