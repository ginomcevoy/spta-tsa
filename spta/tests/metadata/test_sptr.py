import unittest

from spta.region import Region, SpatioTemporalRegionMetadata


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
        self.assertEquals(rmd_1y_1ppd.dataset_filename, 'raw/sp_small_1y_1ppd.npy')

    def test_dataset_filename_1460_4ppd(self):
        rmd_1y_4ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)
        self.assertEquals(rmd_1y_4ppd.dataset_filename, 'raw/sp_small_1y_4ppd.npy')

    def test_distances_filename_365_1ppd(self):
        rmd_1y_1ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 365, 1)
        self.assertEquals(rmd_1y_1ppd.distances_filename, 'raw/distances_sp_small_1y_1ppd.npy')

    def test_distances_filename_1460_4ppd(self):
        rmd_1y_4ppd = SpatioTemporalRegionMetadata('sp_small', Region(40, 50, 50, 60), 1460, 4)
        self.assertEquals(rmd_1y_4ppd.distances_filename, 'raw/distances_sp_small_1y_4ppd.npy')
