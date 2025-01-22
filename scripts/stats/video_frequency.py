import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
import tqdm

def get_result_paths(result_dir_path, result_filename='results.parquet.gzip', minute=None, hour=None):
    for dir_path, dir_names, filenames in os.walk(result_dir_path):
        for filename in filenames:
            if filename == result_filename:
                file_hour, file_minute = map(int, dir_path.split('/')[-3:-1])
                if hour is not None and file_hour != hour:
                    continue
                if minute is not None and file_minute != minute:
                    continue
                result_path = os.path.join(dir_path, filename)
                yield result_path

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    use = '1hour'
    if use == 'all':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", 'all')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd'))
    elif use == '24hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '24hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', minute=42))
    elif use == '1hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '1hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', hour=19))

    result_paths = sorted(result_paths)

    os.makedirs(output_dir_path, exist_ok=True)

    # result_paths = result_paths[:5]
    video_df = None
    val_count_dfs = None
    for result_path in tqdm.tqdm(result_paths):
        batch_df = pl.read_parquet(result_path)
        if video_df is not None:
            video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
        else:
            video_df = batch_df

    video_df = video_df.with_columns(pl.from_epoch('createTime'))

    # look at change in description
    frequent_poster_df = video_df.group_by('authorUniqueId').len().sort('len').filter(pl.col('len') > 1000).filter(pl.col('authorUniqueId').is_not_null())

    frequent_poster_video_df = video_df.join(frequent_poster_df, on='authorUniqueId', how='inner')

    pass


if __name__ == "__main__":
    main()