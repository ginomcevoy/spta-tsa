'''Unit tests for spta.dataset.base module.'''

import numpy as np
import os
import tempfile
import unittest

from spta.tests.stub import stub_region, stub_dataset

from spta.dataset import base, metadata


class TestFileDataset(unittest.TestCase):
    '''Unit tests for spta.dataset.base.FileDataset.'''

    def setUp(self):
        # assumed to be the dataset of StubFileDataset...
        self.input_dataset = stub_region.numpy_3d_4spd_stub()

    def test_retrieve_no_conversion(self):
        # given a stub dataset and a temporal directory
        with tempfile.TemporaryDirectory() as temp_dir:

            time_to_series = metadata.SamplesPerDay(4)
            dataset_temporal_md = metadata.TemporalMetadata(2014, 2015, time_to_series)
            dataset = stub_dataset.StubFileDataset(dataset_temporal_md, temp_dir)

            # when trying to retrieve the whole dataset
            temporal_md = dataset_temporal_md
            result = dataset.retrieve(temporal_md)

            # then we get the same dataset as the input
            expected = self.input_dataset
            stub_region.verify_result_is_expected(self, result, expected)

            # then a file was saved as cache
            expected_filename = 'test_2014_2015_4spd.npy'
            expected_filepath = os.path.join(temp_dir, expected_filename)
            self.assertTrue(os.path.isfile(expected_filepath))

            # then this file contains the data
            cached_dataset = np.load(expected_filepath)
            stub_region.verify_result_is_expected(self, cached_dataset, expected)

    def test_retrieve_converted(self):
        # given a stub dataset and a temporal directory
        with tempfile.TemporaryDirectory() as temp_dir:

            time_to_series = metadata.SamplesPerDay(4)
            dataset_temporal_md = metadata.TemporalMetadata(2014, 2015, time_to_series)
            dataset = stub_dataset.StubFileDataset(dataset_temporal_md, temp_dir)

            # when trying to retrieve 1 sample per day
            temporal_md = metadata.TemporalMetadata(2014, 2015, metadata.SamplesPerDay(1))
            result = dataset.retrieve(temporal_md)

            # then we get an averaged dataset
            expected = np.empty((1, 2, 3))
            expected[:, 0, 0] = np.array((3,))
            expected[:, 0, 1] = np.array((8,))
            expected[:, 0, 2] = np.array((13,))
            expected[:, 1, 0] = np.array((18,))
            expected[:, 1, 1] = np.array((23,))
            expected[:, 1, 2] = np.array((28,))
            stub_region.verify_result_is_expected(self, result, expected)

            # then a file was saved as cache
            expected_filename = 'test_2014_2015_1spd.npy'
            expected_filepath = os.path.join(temp_dir, expected_filename)
            self.assertTrue(os.path.isfile(expected_filepath))

            # then this file contains the averaged data
            cached_dataset = np.load(expected_filepath)
            stub_region.verify_result_is_expected(self, cached_dataset, expected)


class TestCreateDatasetInstance(unittest.TestCase):
    '''Unit tests for spta.dataset.base.create_dataset_instance function.'''

    def test_create_dataset_instance(self):

        # given a temporal directory
        with tempfile.TemporaryDirectory() as temp_dir:

            time_to_series = metadata.SamplesPerDay(4)
            dataset_temporal_md = metadata.TemporalMetadata(2014, 2015, time_to_series)
            kwargs = {'dataset_temporal_md': dataset_temporal_md, 'temp_dir': temp_dir}

            # when creating the instance using the class name
            full_class_name = 'spta.tests.stub.stub_dataset.StubFileDataset'
            instance = base.create_dataset_instance(full_class_name, **kwargs)

            # then we get a proper instance of StubFileDataset
            numpy_dataset = instance.retrieve(dataset_temporal_md)
            stub_region.verify_result_is_expected(self, numpy_dataset, stub_region.numpy_3d_4spd_stub())
