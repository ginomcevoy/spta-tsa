# FIXME this doesn't work anymore!

import numpy as np
from matplotlib import pyplot as plt

from spta.region.mask import MaskRegionFuzzy
from spta.util import log as log_util
from .forecast import ArimaForecasting

# default forecast length
FORECAST_LENGTH = 8

# determines how many times to evaluate error
THRESHOLD_STEPS = 12


class ArimaFuzzySensibility(log_util.LoggerMixin):
    '''
    Given a single fuzzy spatio-temporal cluster, and (p, d, q) ARIMA hyper-parameters,
    perform the following analysis:

      - Train an ARIMA model at the centroid

      - Vary the value of the threshold, sweeping from T = 0 to T = 1, then evaluate the
        RMSE of the MASE error each time.

      - Plot the RMSE errors against T.
    '''

    def __init__(self, arima_params):

        # delegate ARIMA tasks to this implementation
        self.arima_params = arima_params
        self.arima_forecasting = ArimaForecasting(self.arima_params, None)

    def plot_error_vs_threshold(self, spt_cluster, threshold_max=1, forecast_len=FORECAST_LENGTH,
                                threshold_steps=THRESHOLD_STEPS, error_type='MASE'):

        # sanity check
        assert hasattr(spt_cluster, 'mask_region')
        assert isinstance(spt_cluster.mask_region, MaskRegionFuzzy)

        errors_by_threshold = np.empty(threshold_steps)
        step = threshold_max / threshold_steps
        index = 0

        for threshold in np.arange(0, threshold_max, step):

            log_msg = 'Analyzing {} with {} for threshold T={}'
            self.logger.info(log_msg.format(spt_cluster, self.arima_params, threshold))

            # fix the threshold that will affect cluster membership
            # as threshold increases, the cluster can start adding new members
            spt_cluster.mask_region.threshold = threshold

            # train from scratch
            self.arima_forecasting.train_models(spt_cluster, forecast_len)

            # forecast using cluster centroid (medoid)
            error_region = self.arima_forecasting.\
                forecast_whole_region_with_single_model(spt_cluster.centroid, forecast_len,
                                                        error_type)

            # RMSE error to get a single error value for entire cluster
            errors_by_threshold[index] = error_region.overall_error

            log_msg = 'RMSE of forecast MASE error with T={} -> {:.3f}'
            self.logger.debug(log_msg.format(threshold, errors_by_threshold[index]))

            index += 1

        # plot threshold vs error
        plt.plot(np.arange(0, threshold_max, step), errors_by_threshold, 'bo')
        plt.title('RMSE of forecast MASE error, {}, {}'.format(spt_cluster, self.arima_params))
        plt.xlabel('Threshold')
        plt.ylabel('RMSE')
        plt.ylim(0, np.max(errors_by_threshold) * 1.1)
        plt.grid(True)
        plt.show()
