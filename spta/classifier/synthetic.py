import csv
import random

from spta.dataset import synthetic_temporal


def generate_synthetic_series(clustering_k_min, clustering_k_max, clustering_seeds, num_entries,
                              series_len, random_seed, a_weight=0.01, b_min=298, b_max=302, noise=0.2):

    random.seed(random_seed)
    available_ks = range(clustering_k_min, clustering_k_max + 1)

    output = []

    # iterates for each random synthetic tuple to be created
    for entry in range(0, num_entries):

        # this completes the clustering configuration
        entry_k = random.choice(available_ks)
        entry_seed = random.choice(clustering_seeds)

        # cluster_index is bounded to [0, k>
        entry_cluster_index = random.choice(range(0, entry_k))

        # the synthetic data is a line where the slope is a function of the parameters
        def slope(entry_k, entry_seed, entry_cluster_index, a_weight):

            # either +1 or -1
            pos_or_neg = (entry_seed % 2) * 2 - 1

            # a positive or negative slope weighted by a_weight
            return (entry_k - entry_cluster_index) * pos_or_neg * a_weight

        a = slope(entry_k, entry_seed, entry_cluster_index, a_weight)
        series = synthetic_temporal.synthetic_temperature(series_len, a, b_min, b_max, noise)

        output_entry = {
            'k': entry_k,
            'seed': entry_seed,
            'cluster_index': entry_cluster_index,
            'series': series
        }

        output.append(output_entry)

    return output


def generate_synthetic_csv(clustering_k_min, clustering_k_max, clustering_seeds, num_entries,
                           series_len, random_seed, csv_filename,
                           region_id='whole_real_brazil_2014_2014_1spd', clustering_type='kmedoids',
                           clustering_mode='lite', a_weight=0.01, b_min=298, b_max=302, noise=0.2):

    synthetic_series = generate_synthetic_series(clustering_k_min, clustering_k_max, clustering_seeds, num_entries,
                                                 series_len, random_seed, a_weight, b_min, b_max, noise)

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        # header
        header = ['region_id', 'type', 'k', 'seed', 'mode', 'cluster_index']
        series_header = [
            's' + str(i)
            for i in range(0, series_len)
        ]
        header.extend(series_header)
        csv_writer.writerow(header)

        for entry in synthetic_series:
            row = [region_id, clustering_type, entry['k'], entry['seed'], clustering_mode, entry['cluster_index']]
            series_row = entry['series']
            series_row_str = [
                '{:0.3f}'.format(elem)
                for elem in series_row
            ]
            row.extend(series_row_str)
            csv_writer.writerow(row)


if __name__ == '__main__':
    clustering_k_min = 2
    clustering_k_max = 52
    clustering_seeds = (0, )
    num_entries = 5000
    series_len = 365
    random_seed = 0
    csv_filename = 'outputs/test.csv'
    generate_synthetic_csv(clustering_k_min, clustering_k_max, clustering_seeds, num_entries,
                           series_len, random_seed, csv_filename)
