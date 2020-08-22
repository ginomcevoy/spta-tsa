import unittest

from spta.region import Point, Region
from spta.region.metadata import SpatioTemporalRegionMetadata


class TestSptrMetadata(unittest.TestCase):
    '''
    Unit tests for metadata.SpatioTemporalRegionMetadata
    '''

    # def __init__(self, name, region, year_start, year_end, ppd, centroid=None,
    #              scaled=True, dataset_dir='raw'):

    def setUp(self):
        # name, region
        self.name_region = ('sp_small', Region(40, 50, 50, 60))

    def test_dataset_filename_2015_2015_1spd(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1)
        self.assertEquals(metadata.dataset_filename, 'raw/sp_small_2015_2015_1spd_scaled.npy')

    def test_dataset_filename_2015_2015_4spd(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 4)
        self.assertEquals(metadata.dataset_filename, 'raw/sp_small_2015_2015_4spd_scaled.npy')

    def test_dataset_filename_2015_2015_1spd_not_scaled(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=False)
        self.assertEquals(metadata.dataset_filename, 'raw/sp_small_2015_2015_1spd.npy')

    def test_dataset_filename_2014_2015_1spd(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2014, 2015, 1)
        self.assertEquals(metadata.dataset_filename, 'raw/sp_small_2014_2015_1spd_scaled.npy')

    def test_distances_filename_2015_2015_1spd(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1)

        expected = 'raw/distances_sp_small_2015_2015_1spd_scaled.npy'
        self.assertEquals(metadata.distances_filename, expected)

    def test_scaled_min_filename_2015_2015_1spd_scaled(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        expected = 'raw/sp_small_2015_2015_1spd_scaled_min.npy'
        self.assertEquals(metadata.scaled_min_filename, expected)

    def test_scaled_max_filename_2015_2015_1spd_scaled(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        expected = 'raw/sp_small_2015_2015_1spd_scaled_max.npy'
        self.assertEquals(metadata.scaled_max_filename, expected)

    def test_pickle_filename_2015_2015_1spd_scaled(self):
        metadata = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        expected = 'pickle/sp_small_2015_2015_1spd_scaled/sp_small_2015_2015_1spd_scaled.pickle'
        self.assertEquals(metadata.pickle_filename(), expected)

    def test_index_to_absolute_point_index_0(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        # when
        point = sp_small_md.index_to_absolute_point(0)

        # then
        self.assertEquals(point, Point(40, 50))

    def test_index_to_absolute_point_index_1(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        # when
        point = sp_small_md.index_to_absolute_point(1)

        # then
        self.assertEquals(point, Point(40, 51))

    def test_index_to_absolute_point_index_85(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)

        # when
        point = sp_small_md.index_to_absolute_point(85)

        # then 85 -> (8, 5) in subregion
        self.assertEquals(point, Point(48, 55))

    def test_output_dir(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata(*self.name_region, 2015, 2015, 1, scaled=True)
        output_home = 'outputs'

        # when
        output_dir = sp_small_md.output_dir(output_home)

        # then
        self.assertEqual(output_dir, 'outputs/sp_small_2015_2015_1spd_scaled')
