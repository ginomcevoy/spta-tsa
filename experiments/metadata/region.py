from spta.region import Region
from spta.region.temporal import SpatioTemporalRegionMetadata


def predefined_regions():
    '''
    Add new regions here, and use an identifier.
    Metadata: <region name> <region> <series length> <points per day> <use last years?>

    The name is used by default to find the silhouette analysis
    '''
    region_metadata = {

        'brian_1y_1ppd': SpatioTemporalRegionMetadata(
            'brian', Region(40, 80, 45, 85), series_len=365, ppd=1, last=True, normalized=False),

        'midregion_1y_1ppd': SpatioTemporalRegionMetadata(
            'midregion', Region(20, 60, 25, 65), series_len=365, ppd=1, last=True,
            normalized=False),

        'nordeste_small_1y_1ppd': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), series_len=365, ppd=1, last=True,
            normalized=False),

        'nordeste_small_1y_1ppd_norm': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), series_len=365, ppd=1, last=True,
            normalized=True),

        'sp_small': SpatioTemporalRegionMetadata(
            'sp_small', Region(40, 50, 50, 60), series_len=1460, ppd=4, last=True,
            normalized=False),

        'sp_rj_1y_1ppd_first': SpatioTemporalRegionMetadata(
            'sp_rj', Region(40, 75, 50, 85), series_len=365, ppd=1, last=False, normalized=False),

        'whole_brazil_1y_1ppd': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), series_len=365, ppd=1, last=True,
            normalized=False),

        'whole_brazil_1y_1ppd_norm': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), series_len=365, ppd=1, last=True,
            normalized=True)
    }

    return region_metadata
