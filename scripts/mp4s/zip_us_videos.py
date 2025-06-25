import configparser
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

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
    bytes_dir_paths = config['paths']['mp4_paths'].split(',')

    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    for use in ['24hour', '1hour']:
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
        for result_path in tqdm(result_paths):
            batch_df = pl.read_parquet(result_path, columns=['id', 'locationCreated', 'video'])\
                .filter(pl.col('locationCreated') == 'US')\
                .filter(pl.col('video').struct.field('duration') > 0)\
                .select(['id'])
            if video_df is not None:
                video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
            else:
                video_df = batch_df

        print(f"Found {len(video_df)} videos for {use}")

        video_df = video_df.with_columns([
            pl.col('id').cast(pl.UInt64)
                .map_elements(lambda i: format(i, '064b'), pl.String)
                .str.slice(0, 32)
                .map_elements(lambda s: int(s, 2), pl.UInt64)
                .alias('timestamp'),
            pl.lit(bytes_dir_paths).alias('bytes_dir_paths'),
        ])
        video_df = video_df.explode('bytes_dir_paths').rename({'bytes_dir_paths': 'bytes_dir_path'})
        video_df = video_df.with_columns(
            pl.concat_str([
                pl.col('bytes_dir_path'),
                pl.col('timestamp').cast(pl.String),
                pl.lit('/'),
                pl.col('id').cast(pl.String),
                pl.lit('.mp4'),
            ]).alias('video_path')
        )
        video_df = video_df.with_columns(pl.col('video_path').map_elements(os.path.exists, return_dtype=pl.Boolean, strategy='threading').alias('video_path_exists'))

        # get videos that have a video path that exists
        # and get that video pat
        video_df = video_df.filter(pl.col('video_path_exists'))

        print(f"Found {len(video_df)} files for {use}")

        video_df = video_df.with_columns(pl.col('video_path').map_elements(os.path.getsize, return_dtype=pl.UInt64, strategy='threading').alias('video_path_bytes'))

        total_gbs = video_df['video_path_bytes'].sum() / 1e9
        print(f"Total {total_gbs} GBs")

        zip_file_path = f"./data/{use}_us_videos_sample_100.zip"
        os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
        with zipfile.ZipFile(zip_file_path, mode='w') as f:
            for file_path in video_df.sample(100)['video_path'].to_list():
                f.write(file_path)


if __name__ == '__main__':
    main()