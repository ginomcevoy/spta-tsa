import unittest

from spta.region import Point, Region
from spta.region.metadata import SpatioTemporalRegionMetadata
from spta.dataset.metadata import TemporalMetadata, SamplesPerDay


class TestSptrMetadata(unittest.TestCase):
    '''
    Unit tests for metadata.SpatioTemporalRegionMetadata
    '''

    # def __init__(self, name, region, temporal_md, dataset_class_name, centroid=None,
    #              scaled=True, dataset_dir='raw', **dataset_kwargs):

    def setUp(self):
        # name, region
        self.name_region = ('sp_small', Region(40, 50, 50, 60))

        # use a known stub dataset
        self.class_name = 'spta.tests.stub.stub_dataset.StubFileDataset'
        dataset_temporal_md = TemporalMetadata(1979, 2015, SamplesPerDay(4))
        self.dataset_kwargs = {'dataset_temporal_md': dataset_temporal_md, 'temp_dir': '/tmp'}

    def test_dataset_filename_2015_2015_1spd(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)
        self.assertEqual(metadata.dataset_filename, 'raw/sp_small_2015_2015_1spd_scaled.npy')

    def test_dataset_filename_2015_2015_4spd(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(4))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)
        self.assertEqual(metadata.dataset_filename, 'raw/sp_small_2015_2015_4spd_scaled.npy')

    def test_dataset_filename_2015_2015_1spd_not_scaled(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, scaled=False, **self.dataset_kwargs)
        self.assertEqual(metadata.dataset_filename, 'raw/sp_small_2015_2015_1spd.npy')

    def test_dataset_filename_2014_2015_1spd(self):
        temporal_md = TemporalMetadata(2014, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)
        self.assertEqual(metadata.dataset_filename, 'raw/sp_small_2014_2015_1spd_scaled.npy')

    def test_distances_filename_2015_2015_1spd(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        expected = 'raw/distances_sp_small_2015_2015_1spd_scaled.npy'
        self.assertEqual(metadata.distances_filename, expected)

    def test_scaled_min_filename_2015_2015_1spd_scaled(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        expected = 'raw/sp_small_2015_2015_1spd_scaled_min.npy'
        self.assertEqual(metadata.scaled_min_filename, expected)

    def test_scaled_max_filename_2015_2015_1spd_scaled(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        expected = 'raw/sp_small_2015_2015_1spd_scaled_max.npy'
        self.assertEqual(metadata.scaled_max_filename, expected)

    def test_pickle_filename_2015_2015_1spd_scaled(self):
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        expected = 'pickle/sp_small_2015_2015_1spd_scaled/sp_small_2015_2015_1spd_scaled.pickle'
        self.assertEqual(metadata.pickle_filename(), expected)

    def test_index_to_absolute_point_index_0(self):
        # given
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        # when
        point = metadata.index_to_absolute_point(0)

        # then
        self.assertEqual(point, Point(40, 50))

    def test_index_to_absolute_point_index_1(self):
        # given
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        # when
        point = metadata.index_to_absolute_point(1)

        # then
        self.assertEqual(point, Point(40, 51))

    def test_index_to_absolute_point_index_85(self):
        # given
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)

        # when
        point = metadata.index_to_absolute_point(85)

        # then 85 -> (8, 5) in subregion
        self.assertEqual(point, Point(48, 55))

    def test_output_dir(self):
        # given
        temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
        metadata = SpatioTemporalRegionMetadata(*self.name_region, temporal_md, self.class_name, **self.dataset_kwargs)
        output_home = 'outputs'

        # when
        output_dir = metadata.output_dir(output_home)

        # then
        self.assertEqual(output_dir, 'outputs/sp_small_2015_2015_1spd_scaled')
