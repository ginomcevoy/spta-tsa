'''
Execute this program to perform ARIMA forecasting and error evaluation on an entire region.
'''
import argparse

from spta.arima.train import TrainerArimaPDQ
from spta.model.forecast import ForecastAnalysis
from spta.distance.dtw import DistanceByDTW
from spta.util import log as log_util

from experiments.metadata.arima import predefined_arima_suites, arima_suite_by_name
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
    arima_options = predefined_arima_suites().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('arima', help='Name of the ARIMA experiment', choices=arima_options)
    parser.add_argument('--parallel', help='number of parallel workers')
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    do_arima_forecast(args)


def do_arima_forecast(args):

    # get the region from metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = spt_region_metadata.create_instance()

    _, x_len, y_len = spt_region.shape

    # try to get a pre-calculated centroid, otherwise calculate it
    # TODO allow other distances
    spt_region.centroid = centroid_by_region_and_distance(args.region, DistanceByDTW())

    # use parallelization?
    parallel_workers = None
    if args.parallel:
        parallel_workers = int(args.parallel)

    # get the ARIMA suite
    arima_suite = arima_suite_by_name(args.arima)
    arima_results = {}
    for arima_params in arima_suite.arima_params_gen():

        # do the analysis with current ARIMA hyper-parameters, only save errors
        # forecasting_pdq = ArimaForecastingPDQ(arima_params, parallel_workers=parallel_workers)

        arima_trainer = TrainerArimaPDQ(arima_params, x_len, y_len)
        forecast_analysis = ForecastAnalysis(arima_trainer, parallel_workers)
        overall_errors, _, _ = forecast_analysis.analyze_errors(spt_region, 'MASE')

        # ArimaErrors = namedtuple('ArimaErrors', ('minimum', 'min_local', 'centroid', 'maximum'))
        arima_results[arima_params] = overall_errors

    # print results
    for (arima_params, overall_errors) in arima_results.items():

        result_line = '{} errors -> each={:.2f}, min_local={:.2f}, centroid={:.2f}'
        print(result_line.format(arima_params, overall_errors.each, overall_errors.min_local,
                                 overall_errors.centroid))


if __name__ == '__main__':
    processRequest()
