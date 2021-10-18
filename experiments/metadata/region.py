from spta.dataset.metadata import TemporalMetadata, SamplesPerDay, AveragePentads
from spta.region import Region
from spta.region.metadata import SpatioTemporalRegionMetadata


def predefined_regions():
    '''
    Add new regions here, and use an identifier.
    Metadata: <region name> <region> <series length> <points per day> <use last years?>

    The name is used by default to find the silhouette analysis
    '''

    # CSFR dataset
    temp_md_2011_2015_1spd = TemporalMetadata(2011, 2015, SamplesPerDay(1))
    temp_md_2013_2013_1spd = TemporalMetadata(2013, 2013, SamplesPerDay(1))
    temp_md_2014_2014_1spd = TemporalMetadata(2014, 2014, SamplesPerDay(1))
    temp_md_2015_2015_1spd = TemporalMetadata(2015, 2015, SamplesPerDay(1))
    csfr = 'spta.dataset.csfr.DatasetCSFR'

    # CHIRPS dataset
    temp_md_avg_pentads = TemporalMetadata(2010, 2018, AveragePentads())
    chirps2D = 'spta.dataset.chirps.DatasetCHIRPS2D'
    chirps3D = 'spta.dataset.chirps.DatasetCHIRPS3D'
    chirps2D_region = Region(0, 23604, 0, 1)
    chirps3D_region = Region(0, 200, 0, 338)

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

        'chirps_2D_2010_2018_avg_pentads': SpatioTemporalRegionMetadata(
            'chirps2d', chirps2D_region, temp_md_avg_pentads, chirps2D, scaled=False),

        'chirps_2D_2010_2018_avg_pentads_scaled': SpatioTemporalRegionMetadata(
            'chirps2d', chirps2D_region, temp_md_avg_pentads, chirps2D, scaled=True),

        'chirps_3D_2010_2018_avg_pentads': SpatioTemporalRegionMetadata(
            'chirps3d', chirps3D_region, temp_md_avg_pentads, chirps3D, scaled=False),

        'chirps_3D_2010_2018_avg_pentads_scaled': SpatioTemporalRegionMetadata(
            'chirps3d', chirps3D_region, temp_md_avg_pentads, chirps3D, scaled=True),
    }

    return region_metadata
