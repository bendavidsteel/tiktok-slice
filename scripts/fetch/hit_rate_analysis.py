import json
import os

import matplotlib.pyplot as plt
import polars as pl
import tqdm


def main():
    csv_path = './data/hit_rates.csv'
    if not os.path.exists(csv_path):
        result_df = pl.scan_parquet('./data/stats/all/hit_rate.parquet.zstd')
        result_df = result_df.with_columns(
            pl.col('args').cast(pl.UInt64)
                            .map_elements(lambda i: format(i, '064b'), pl.String)
                            .str.slice(42, 64)
                            .map_elements(lambda s: int(s, 2), pl.UInt64)
                            .alias('final_bits')
        )

        print(f"Getting num hits")
        num_hits = result_df.filter(pl.col('success')).collect().shape[0]

        print("Getting most common")
        most_common_df = result_df.filter(pl.col('success')).collect()['final_bits'].value_counts().sort('count', descending=True)
        most_common_df = most_common_df.with_columns(
            pl.col('count').cum_sum().alias('cumsum')
        )
        total = most_common_df['count'].sum()

        hit_rate = []
        reduced_hits = []
        shares = [0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
        for share in shares:
            print(f"Getting share: {share}")
            threshold = int(total * share)
            filtered_df = most_common_df.filter(pl.col('cumsum') <= threshold)
            filtered_result_df = result_df.filter(pl.col('final_bits').is_in(filtered_df['final_bits']))
            num_reduced_hits = filtered_result_df.filter(pl.col('success')).select(pl.len()).collect().item()
            num_requests = filtered_result_df.select(pl.len()).collect().item()
            reduced_hits.append(num_reduced_hits / num_hits)
            hit_rate.append(num_reduced_hits / num_requests)

        pl.DataFrame({'hit_rate': hit_rate, 'reduced_hits': reduced_hits}).write_csv()
    else:
        df = pl.read_csv(csv_path)
        hit_rate = df['hit_rate'].to_list()
        reduced_hits = df['reduced_hits'].to_list()

    rates = [128, 64, 32]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(hit_rate, reduced_hits, marker='o')
    ax.set_xticks([1 / i for i in rates], labels=[f'1/{i}' for i in rates])
    ax.set_xlabel('Success rate of ID generation')
    ax.set_ylabel('Pct. of videos fetched')
    fig.savefig('./figs/hit_rate_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
