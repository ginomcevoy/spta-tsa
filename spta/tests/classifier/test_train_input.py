'''
Unit tests for spta.classifier.train_input module.
'''

from spta.region import Region, Point
from spta.region.metadata import SpatioTemporalRegionMetadata

from spta.arima import AutoArimaParams
from spta.classifier.train_input import MedoidsChoiceMinDistance, MedoidsChoiceMinPredictionError
from spta.distance.dtw import DistanceByDTW

from spta.tests.stub import stub_clustering

import unittest


class TestMedoidsChoiceMinDistance(unittest.TestCase):
    '''
    Unit tests for train_input.MedoidsChoiceMinDistance class.
    '''

    def setUp(self):
        self.region_metadata = SpatioTemporalRegionMetadata('nordeste_small',
                                                            Region(43, 50, 85, 95), 2015, 2015, 1,
                                                            scaled=False)
        self.distance_measure = DistanceByDTW()

    def test_choose_medoid(self):
        # NOTE this requires a pre-calculated distance matrix!

        # given
        suite_result = {
            'kmedoids_k2_seed0_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k2_seed1_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k3_seed0_lite': [Point(45, 86), Point(48, 89), Point(45, 92)],
            'kmedoids_k3_seed1_lite': [Point(45, 86), Point(45, 92), Point(48, 89)]
        }
        point = Point(5, 5)

        # given no threshold
        threshold = 0
        medoid_histogram = {}

        # when
        instance = MedoidsChoiceMinDistance(region_metadata=self.region_metadata,
                                            distance_measure=self.distance_measure)
        result = instance.choose_medoid(suite_result, point, threshold, medoid_histogram)

        # then
        self.assertEqual(result, ('kmedoids_k3_seed0_lite', 1, Point(48, 89)))

    def test_choose_medoid_but_threshold_was_reached(self):

        # given
        suite_result = {
            'kmedoids_k2_seed0_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k2_seed1_lite': [Point(45, 86), Point(47, 91)],
            'kmedoids_k3_seed0_lite': [Point(45, 86), Point(48, 89), Point(45, 92)],
            'kmedoids_k3_seed1_lite': [Point(45, 86), Point(45, 92), Point(48, 89)]
        }
        point = Point(5, 5)

        # given that the threshold was reached for the medoid that would have been otherwise chosen
        threshold = 1
        medoid_histogram = {('kmedoids_k3_seed0_lite', 1): 1}

        # when
        instance = MedoidsChoiceMinDistance(region_metadata=self.region_metadata,
                                            distance_measure=self.distance_measure)
        result = instance.choose_medoid(suite_result, point, threshold, medoid_histogram)

        # then
        self.assertEqual(result, ('kmedoids_k3_seed1_lite', 2, Point(48, 89)))

    def test_csv_filepath(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()
        output_home = 'outputs'

        # given input for <count> random points
        count = 10
        random_seed = 0

        instance = MedoidsChoiceMinDistance(region_metadata=self.region_metadata,
                                            distance_measure=self.distance_measure)

        # when
        result = instance.csv_filepaths(output_home, kmedoids_suite, count, random_seed)

        # then
        expected = (
            'outputs/nordeste_small_2015_2015_1spd/dtw/random_point_dist_medoid__kmedoids-quick_count10_seed0.csv',
            'outputs/nordeste_small_2015_2015_1spd/dtw/hist_dist_medoid__kmedoids-quick_count10_seed0.csv'
        )
        self.maxDiff = None
        self.assertEqual(result, expected)


class TestMedoidsChoiceMinPredictionError(unittest.TestCase):
    '''
    Unit tests for train_input.MedoidsChoiceMinPredictionError class.
    '''

    def setUp(self):
        self.region_metadata = SpatioTemporalRegionMetadata('nordeste_small',
                                                            Region(43, 50, 85, 95), 2015, 2015, 1,
                                                            scaled=False)
        self.distance_measure = DistanceByDTW()

    def test_csv_filepath(self):

        # given
        kmedoids_suite = stub_clustering.kmedoids_quick_stub()
        output_home = 'outputs'

        # given input for <count> random points
        count = 10
        random_seed = 0

        # given some solver metadata
        model_params = AutoArimaParams(1, 1, 3, 3, None, True)
        test_len = 8
        error_type = 'sMAPE'

        instance = MedoidsChoiceMinPredictionError(region_metadata=self.region_metadata,
                                                   distance_measure=self.distance_measure,
                                                   model_params=model_params,
                                                   test_len=test_len,
                                                   error_type=error_type,
                                                   output_home=output_home)

        # when
        result = instance.csv_filepaths(output_home, kmedoids_suite, count, random_seed)

        # then
        expected = (
            'outputs/nordeste_small_2015_2015_1spd/dtw/random_point_medoid_min_pred_error__auto-arima-start_p1-start_q1-max_p3-max_q3-dNone-stepwiseTrue__tp8__sMAPE__kmedoids-quick_count10_seed0.csv',
            'outputs/nordeste_small_2015_2015_1spd/dtw/hist_medoid_min_pred_error__auto-arima-start_p1-start_q1-max_p3-max_q3-dNone-stepwiseTrue__tp8__sMAPE__kmedoids-quick_count10_seed0.csv'
        )
        self.maxDiff = None
        self.assertEqual(result, expected)

    def test_build_solver_metadata(self):

        # given
        clustering_repr = 'kmedoids_k3_seed0_lite'
        output_home = 'outputs'

        # given some solver metadata
        model_params = AutoArimaParams(1, 1, 3, 3, None, True)
        test_len = 8
        error_type = 'sMAPE'

        instance = MedoidsChoiceMinPredictionError(region_metadata=self.region_metadata,
                                                   distance_measure=self.distance_measure,
                                                   model_params=model_params,
                                                   test_len=test_len,
                                                   error_type=error_type,
                                                   output_home=output_home)
        solver_metadata = instance.build_solver_metadata(clustering_repr)
        self.assertTrue(solver_metadata is not None)
        self.assertEqual(solver_metadata.error_type, 'sMAPE')
        self.assertEqual(solver_metadata.clustering_metadata.k, 3)
