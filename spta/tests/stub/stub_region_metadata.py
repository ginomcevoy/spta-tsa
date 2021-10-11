
from spta.dataset.metadata import TemporalMetadata, SamplesPerDay
from spta.region import Region
from spta.region.metadata import SpatioTemporalRegionMetadata


def get_stub_region_md():
    '''
    Uses the stub dataset to generate the metadata for a spatio-temporal region,
    The filename would be "raw/test_sptr_2015_2015_1spd.npy", but this file is not meant to be loaded.
    '''
    dataset_class_name = 'spta.tests.stub.stub_dataset.StubFileDataset'
    dataset_temporal_md = TemporalMetadata(1979, 2015, SamplesPerDay(4))
    dataset_kwargs = {'dataset_temporal_md': dataset_temporal_md, 'temp_dir': '/tmp'}
    temporal_md = TemporalMetadata(2015, 2015, SamplesPerDay(1))
    region_metadata = SpatioTemporalRegionMetadata(name='test_sptr',
                                                   region=Region(40, 50, 50, 60),
                                                   temporal_md=temporal_md,
                                                   dataset_class_name=dataset_class_name,
                                                   scaled=False,
                                                   **dataset_kwargs)
    return region_metadata
