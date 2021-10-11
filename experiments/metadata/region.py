from spta.dataset.metadata import TemporalMetadata, SamplesPerDay, AveragePentads
from spta.region import Region
from spta.region.metadata import SpatioTemporalRegionMetadata


def predefined_regions():
    '''
    Add new regions here, and use an identifier.
    Metadata: <region name> <region> <series length> <points per day> <use last years?>

    The name is used by default to find the silhouette analysis
    '''

    temp_md_2011_2015_1spd = TemporalMetadata(2011, 2015, SamplesPerDay(1))
    temp_md_2013_2013_1spd = TemporalMetadata(2013, 2013, SamplesPerDay(1))
    temp_md_2014_2014_1spd = TemporalMetadata(2014, 2014, SamplesPerDay(1))
    temp_md_2015_2015_1spd = TemporalMetadata(2015, 2015, SamplesPerDay(1))
    csfr = 'spta.dataset.csfr.DatasetCSFR'

    temp_md_avg_pentads = TemporalMetadata(1981, 2019, AveragePentads())
    rain = 'spta.dataset.rain.DatasetRain'

    region_metadata = {

        'brian_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'brian', Region(40, 80, 45, 85), temp_md_2015_2015_1spd, csfr, scaled=False),

        'midregion_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'midregion', Region(20, 60, 25, 65), temp_md_2015_2015_1spd, csfr, scaled=False),

        'nordeste_small_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), temp_md_2015_2015_1spd, csfr, scaled=False),

        'nordeste_small_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), temp_md_2015_2015_1spd, csfr, scaled=True),

        'nordeste_small_2014_2014_1spd': SpatioTemporalRegionMetadata(
            'nordeste_small', Region(43, 50, 85, 95), temp_md_2014_2014_1spd, csfr, scaled=False),

        'whole_brazil_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), temp_md_2015_2015_1spd, csfr, scaled=False),

        'whole_brazil_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_brazil', Region(20, 100, 15, 95), temp_md_2015_2015_1spd, csfr, scaled=True),

        'whole_real_brazil_2013_2013_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2013_2013_1spd, csfr, scaled=False),

        'whole_real_brazil_2013_2013_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2013_2013_1spd, csfr, scaled=True),

        'whole_real_brazil_2014_2014_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2014_2014_1spd, csfr, scaled=False),

        'whole_real_brazil_2014_2014_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2014_2014_1spd, csfr, scaled=True),

        'whole_real_brazil_2015_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2015_2015_1spd, csfr, scaled=False),

        'whole_real_brazil_2015_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2015_2015_1spd, csfr, scaled=True),

        'whole_real_brazil_2011_2015_1spd': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2011_2015_1spd, csfr, scaled=False),

        'whole_real_brazil_2011_2015_1spd_scaled': SpatioTemporalRegionMetadata(
            'whole_real_brazil', Region(5, 95, 15, 105), temp_md_2011_2015_1spd, csfr, scaled=True),

        # 'rain_avg_pentads': SpatioTemporalRegionMetadata(
        #     'rain', Region(0, 0, 0, 0), temp_md_avg_pentads, rain, scaled=False)
    }

    return region_metadata
