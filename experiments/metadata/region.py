from spta.region import Region
from spta.region.metadata import SpatioTemporalRegionMetadata


def predefined_regions():
    '''
    Add new regions here, and use an identifier.
    Metadata: <region name> <region> <series length> <points per day> <use last years?>

    The name is used by default to find the silhouette analysis
    '''
    region_metadata = {

        'brian_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'brian', Region(40, 80, 45, 85), year_start=2015, year_end=2015, spd=1, scaled=False),

        'midregion_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'midregion', Region(20, 60, 25, 65), year_start=2015, year_end=2015, spd=1,
            scaled=False),

        'nordeste_small_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), year_start=2015, year_end=2015, spd=1,
            scaled=False),

        'nordeste_small_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), year_start=2015, year_end=2015, spd=1,
            scaled=True),

        'nordeste_small_2014_2014_1spd': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), year_start=2014, year_end=2014, spd=1,
            scaled=False),

        'whole_brazil_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), year_start=2015, year_end=2015, spd=1,
            scaled=False),

        'whole_brazil_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), year_start=2015, year_end=2015, spd=1,
            scaled=True),

        'whole_real_brazil_2013_2013_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2013, year_end=2013, spd=1,
            scaled=False),

        'whole_real_brazil_2013_2013_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2013, year_end=2013, spd=1,
            scaled=True),

        'whole_real_brazil_2014_2014_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2014, year_end=2014, spd=1,
            scaled=False),

        'whole_real_brazil_2014_2014_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2014, year_end=2014, spd=1,
            scaled=True),

        'whole_real_brazil_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2015, year_end=2015, spd=1,
            scaled=False),

        'whole_real_brazil_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2015, year_end=2015, spd=1,
            scaled=True),

        'whole_real_brazil_2011_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2011, year_end=2015, spd=1,
            scaled=False),

        'whole_real_brazil_2011_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), year_start=2011, year_end=2015, spd=1,
            scaled=True)
    }

    return region_metadata
