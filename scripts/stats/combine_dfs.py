import os

import polars as pl
from tqdm import tqdm

def main():
    results_path = './data/results/2024_04_10'
    # walk through all subdirectories and collect DataFrames
    all_dfs = []
    pbar = tqdm(total=60*60 + 23*60)
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                file_path = os.path.join(root, file)
                df = pl.read_parquet(file_path)
                all_dfs.append(df)
                pbar.update(1)

    # Concatenate all DataFrames at once
    combined_df = pl.concat(all_dfs, how='diagonal_relaxed')
    combined_df.write_parquet('./data/results/combined_videos.parquet.zstd', compression='zstd')
    print(f"Combined DataFrame shape: {combined_df.shape}")

if __name__ == "__main__":
    main()