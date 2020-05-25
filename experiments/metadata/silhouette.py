'''
Define silhouette analysis by name, can override metadata
'''
from spta.kmedoids.silhouette import silhouette_default_metadata


def silhouette_metadata_by_name(silhouette_name):
    '''
    Get a silhouette analysis by name. Multiple silhouette analysis can be defined for the same
    region metadata.

    The sptr_name is used to save plots.
    '''

    # add new here
    silhouettes_metadata = {
        'brian': silhouette_default_metadata(
            k_values=range(2, 6), seed_values=range(0, 3), plot_name='brian'),

        'midregion': silhouette_default_metadata(
            k_values=range(2, 8), seed_values=range(0, 3), plot_name='midregion'),

        'nordeste_small': silhouette_default_metadata(
            k_values=range(2, 8), seed_values=range(0, 3), plot_name='nordeste_small'),

        'sp_small': silhouette_default_metadata(
            k_values=(2, 3, 4), seed_values=(0,), plot_name='sp_small'),

        'sp_rj': silhouette_default_metadata(
            k_values=(2, 3, 4), seed_values=(0,), plot_name='sp_rj'),

        'whole_brazil': silhouette_default_metadata(
            k_values=range(12, 13), seed_values=range(0, 4), plot_name='whole_brazil'),
    }

    return silhouettes_metadata[silhouette_name]
