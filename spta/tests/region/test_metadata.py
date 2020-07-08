import unittest

from spta.region import Point, Region
from spta.region.metadata import SpatioTemporalRegionMetadata


class TestSptrMetadata(unittest.TestCase):

    def test_time_str_365_1ppd(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1)
        self.assertEquals(rmd_1y_1ppd.time_str, '1y')

    def test_time_str_730_1ppd(self):
        rmd_2y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 730, 1)
        self.assertEquals(rmd_2y_1ppd.time_str, '2y')

    def test_time_str_730_2ppd(self):
        rmd_1y_2ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 730, 2)
        self.assertEquals(rmd_1y_2ppd.time_str, '1y')

    def test_time_str_1460_1ppd(self):
        rmd_4y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 1)
        self.assertEquals(rmd_4y_1ppd.time_str, '4y')

    def test_time_str_1460_4ppd(self):
        rmd_1y_4ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)
        self.assertEquals(rmd_1y_4ppd.time_str, '1y')

    def test_dataset_filename_365_1ppd(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1)
        self.assertEquals(rmd_1y_1ppd.dataset_filename, 'raw/sp_small_1y_1ppd_norm.npy')

    def test_dataset_filename_1460_4ppd(self):
        rmd_1y_4ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)
        self.assertEquals(rmd_1y_4ppd.dataset_filename, 'raw/sp_small_1y_4ppd_norm.npy')

    def test_dataset_filename_365_1ppd_not_normalized(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=False)
        self.assertEquals(rmd_1y_1ppd.dataset_filename, 'raw/sp_small_1y_1ppd.npy')

    def test_distances_filename_365_1ppd(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1)
        self.assertEquals(rmd_1y_1ppd.distances_filename,
                          'raw/distances_sp_small_1y_1ppd_norm.npy')

    def test_distances_filename_1460_4ppd(self):
        rmd_1y_4ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)
        self.assertEquals(rmd_1y_4ppd.distances_filename,
                          'raw/distances_sp_small_1y_4ppd_norm.npy')

    def test_norm_min_filename_365_1ppd_norm(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)
        self.assertEquals(rmd_1y_1ppd.norm_min_filename, 'raw/sp_small_1y_1ppd_norm_min.npy')

    def test_norm_max_filename_365_1ppd_norm(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)
        self.assertEquals(rmd_1y_1ppd.norm_max_filename, 'raw/sp_small_1y_1ppd_norm_max.npy')

    def test_pickle_filename_365_1ppd_norm(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)
        self.assertEquals(rmd_1y_1ppd.pickle_filename, 'pickle/sp_small_1y_1ppd_norm.pickle')

    def test_index_to_absolute_point_index_0(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)

        # when
        point = sp_small_md.index_to_absolute_point(0)

        # then
        self.assertEquals(point, Point(40, 50))

    def test_index_to_absolute_point_index_1(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)

        # when
        point = sp_small_md.index_to_absolute_point(1)

        # then
        self.assertEquals(point, Point(40, 51))

    def test_index_to_absolute_point_index_85(self):
        # given
        sp_small_md = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1,
                                                   normalized=True)

        # when
        point = sp_small_md.index_to_absolute_point(85)

        # then 85 -> (8, 5) in subregion
        self.assertEquals(point, Point(48, 55))
