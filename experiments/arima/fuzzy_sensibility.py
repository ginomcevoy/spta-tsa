'''
Execute this program to perform fuzzy clustering on a spatio-temporal region, train ARIMA
models,then do fuzzy sensibility analysis for each cluster.
'''
import argparse

from spta.arima.fuzzy_sensibility import ArimaFuzzySensibility
from spta.distance.dtw import DistanceByDTW
from spta.kmedoids import kmedoids_fuzzy
from spta.region.temporal import SpatioTemporalRegion, SpatioTemporalCluster
from spta.util import log as log_util

from experiments.metadata.arima import arima_suite_by_name
from experiments.metadata.arima_clustering import arima_clustering_experiments
from experiments.metadata.region import predefined_regions


def processRequest():

    # parses the arguments
    desc = 'Perform clustering on a spatial-region, ' \
        'then use arima.ArimaFuzzySensibility on each cluster'
    usage = '%(prog)s [-h] <region> <arima_clustering_experiment> [--log=log_level]'
    parser = argparse.ArgumentParser(prog='arima_forecast_cluster', description=desc, usage=usage)

    # for now, need name of region metadata and id of arima_clustering
    region_options = predefined_regions().keys()
    arima_clustering_options = arima_clustering_experiments().keys()
    parser.add_argument('region', help='Name of the region metadata', choices=region_options)
    parser.add_argument('arima_clustering', help='ID of arima clustering experiment',
                        choices=arima_clustering_options)
    parser.add_argument('--fuzzifier', help='integer value for fuzzifier, default = 2',
                        default='2')
    parser.add_argument('--log', help='log level: WARN|INFO|DEBUG')
    parser.add_argument('--plot', help='add the plot of the ARIMA model result at Point(0, 0)?',
                        default=False, action='store_true')

    args = parser.parse_args()
    log_util.setup_log_argparse(args)
    do_arima_fuzzy_sensibility(args)


def do_arima_fuzzy_sensibility(args):

    logger = log_util.logger_for_me(do_arima_fuzzy_sensibility)

    # get the region from metadata
    spt_region_metadata = predefined_regions()[args.region]
    spt_region = SpatioTemporalRegion.from_metadata(spt_region_metadata)

    # get experiment parameters including ARIMA suite
    exp_params = arima_clustering_experiments()[args.arima_clustering]
    (arima_experiment_id, clustering_id, distance_id, k, seed) = exp_params
    arima_suite = arima_suite_by_name(arima_experiment_id)

    # for now, ignore clustering_id and go with K-medoids fuzzy anyway
    m = int(args.fuzzifier)

    # use pre-computed distance matrix
    distance_dtw = DistanceByDTW()
    distance_dtw.load_distance_matrix_2d(spt_region_metadata.distances_filename,
                                         spt_region_metadata.region)

    # faster... (whole_brazil_1y_1ppd, k=8, seed=0)
    # initial_medoids = [5816, 1163, 4295, 4905, 3156, 2648, 172, 3764]
    # initial_medoids = [3764, 2648, 5816, 4905, 4295, 172, 1163, 3156]
    initial_medoids = None

    # build a KmedoidsFuzzyParams object
    kfuzzy_params = kmedoids_fuzzy.\
        kmedoids_fuzzy_default_params(k, m=m, distance_measure=distance_dtw, random_seed=seed,
                                      initial_medoids=initial_medoids)

    # run k-medoids fuzzy
    # KmedoidsResult(k, random_seed, medoids, labels, costs, tot_cost, dist_mat)
    kfuzzy_result = kmedoids_fuzzy.run_kmedoids_fuzzy_from_params(spt_region.as_2d,
                                                                  kfuzzy_params)

    # build the spatio-temporal clusters
    clusters = []
    for i in range(0, k):

        # force threshold = 0 here, we will vary it later
        cluster_i = SpatioTemporalCluster.from_fuzzy_clustering(spt_region, kfuzzy_result.uij,
                                                                cluster_index=i, threshold=0,
                                                                centroids=kfuzzy_result.medoids)
        clusters.append(cluster_i)

    # iterate the spatio-temporal clusters
    for i in range(0, k):

        # the medoid will be used as centroid for the ARIMA analysis
        cluster_i = clusters[i]
        centroid_i = cluster_i.centroid

        logger.info('************************************')
        logger.info('Analyzing cluster {} with medoid: {}'.format(i, centroid_i))
        logger.info('************************************')

        # iterate the ARIMA suite
        for arima_params in arima_suite.arima_params_gen():

            # do the fuzzy sensibility analysis with current ARIMA hyper-parameters
            fuzzy_analysis = ArimaFuzzySensibility(arima_params)
            fuzzy_analysis.plot_error_vs_threshold(cluster_i, threshold_max=0.4, error_type='MASE')


if __name__ == '__main__':
    processRequest()
