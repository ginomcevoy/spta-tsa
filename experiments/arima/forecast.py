'''
Execute this program to perform ARIMA forecasting and error evaluation on an entire region.
'''
import argparse

from spta.arima import arima
from spta.distance.dtw import DistanceByDTW
from spta.region import SpatioTemporalRegion
from spta.util import log as log_util

from experiments.metadata.arima import predefined_arima_experiments
from experiments.metadata.centroid import centroid_by_region_and_distance
from experiments.metadata.region import predefined_regions


def processRequest():

    # parses the arguments
    desc = 'Call arima.evaluate_forecast_errors_arima on a spatio temporal region'
    usage = '%(prog)s [-h] <region> <arima> [--log=log_level]'
    parser = argparse.ArgumentParser(prog='arima_forecast', description=desc, usage=usage)

    # for now, need name of region metadata and the command
    # the silhouette analysis will have the same name as the region metadata
    region_options = predefined_regions().keys()
    arima_options = predefined_arima_experiments().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('arima', help='Name of the ARIMA experiment', choices=arima_options)
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    do_arima_forecast(args)


def do_arima_forecast(args):

    # get the region from metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

    # try to get a pre-calculated centroid, otherwise calculate it
    # TODO allow other distances
    centroid = centroid_by_region_and_distance(args.region, DistanceByDTW())

    # get the ARIMA experiment
    arima_experiment = predefined_arima_experiments()[args.arima]
    for arima_params in arima_experiment:
        (centroid, centroid_arima, training_region, forecast_region, test_region,
            arima_region) = arima.evaluate_forecast_errors_arima(spt_region, arima_params,
                                                                 centroid=centroid)
        arima.plot_one_arima(training_region, forecast_region, test_region, arima_region)


if __name__ == '__main__':
    processRequest()
