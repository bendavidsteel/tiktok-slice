import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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

def extract_video_data(df):
    return df.filter(
        pl.col('return').struct.field('id').is_not_null()
    ).with_columns([
        pl.col('return').struct.field('id').alias('video_id'),
        pl.col('return').struct.field('desc').alias('desc'),
        pl.col('return').struct.field('author').struct.field('uniqueId').alias('authorUniqueId'),
        pl.col('return').struct.field('stats').struct.field('commentCount').cast(pl.Int64).alias('commentCount'),
        pl.col('return').struct.field('stats').struct.field('diggCount').cast(pl.Int64).alias('diggCount'),
        pl.col('return').struct.field('stats').struct.field('shareCount').cast(pl.Int64).alias('shareCount'),
        pl.col('return').struct.field('stats').struct.field('playCount').cast(pl.Int64).alias('playCount'),
        pl.col('return').struct.field('video').struct.field('duration').cast(pl.Int64).alias('videoDuration'),
        pl.col('return').struct.field('imagePost').is_not_null().alias('isImagePost'),
        pl.col('return').map_elements(lambda r: len(r['imagePost']['images']) if 'imagePost' in r and r['imagePost'] else 0, return_dtype=pl.Int32).alias('numImages'),
        pl.col('return').struct.field('locationCreated').alias('locationCreated')
    ]).select(['video_id', 'desc', 'authorUniqueId', 'commentCount', 'diggCount', 'shareCount', 'playCount', 'videoDuration', 'isImagePost', 'numImages', 'locationCreated'])

def extract_error_data(df):
    return df.filter(
        pl.col('return').struct.field('statusCode').is_not_null()
    ).with_columns([
        pl.col('return').struct.field('statusCode').cast(pl.Int32).alias('statusCode'),
        pl.col('return').struct.field('statusMsg').alias('statusMsg')
    ]).select(['statusCode', 'statusMsg'])

def update_value_counts(df, column, existing_counts):
    new_counts = df.group_by(column).count()
    
    if existing_counts is not None:
        updated_counts = pl.concat([existing_counts, new_counts]).group_by(column).sum().sort('count', descending=True)
    else:
        updated_counts = new_counts.sort('count', descending=True)
    
    return updated_counts

def process_batch(result_path, time_count_df):
    batch_result_df = pl.read_parquet(result_path, columns=['result', 'args'])
    batch_result_df = batch_result_df.select([
            'args', 
            pl.col('result').struct.field('return').struct.field('statusMsg'),
            pl.col('result').struct.field('return').struct.field('id')
        ]).with_columns([
            pl.col('args').cast(pl.UInt64),
                ((pl.col('statusMsg') != "item doesn't exist")\
                | pl.col('id').is_not_null()).alias('success')
        ])\
        .select(['args', 'success'])\
        .filter(pl.col('success'))\
        .with_columns(
            pl.from_epoch(pl.col('args').cast(pl.UInt64)
                        .map_elements(lambda i: format(i, '064b'), pl.String)
                        .str.slice(0, 32)
                        .map_elements(lambda s: int(s, 2), pl.UInt64))
                        .alias('createTime')
        )\
        .select(['createTime'])

    # Update value counts
    time_count_df = update_value_counts(batch_result_df, 'createTime', time_count_df)

    return time_count_df


def load_and_write(result_paths, output_dir_path):
    result_paths = sorted(result_paths)
    os.makedirs(output_dir_path, exist_ok=True)
    # result_paths = result_paths[:5]
    video_df = None
    time_count_df = None
    for result_path in tqdm.tqdm(result_paths):
        time_count_df = process_batch(result_path, time_count_df)

    # Finalize results
    time_count_df.write_csv(os.path.join(output_dir_path, "time_counts.csv"))

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '24hour')
    result_paths = list(get_result_paths(base_result_path, result_filename='results.parquet.gzip', minute=42))
    load_and_write(result_paths, output_dir_path)

    output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '1hour')
    result_paths = list(get_result_paths(base_result_path, result_filename='results.parquet.gzip', hour=19))
    load_and_write(result_paths, output_dir_path)
    

if __name__ == "__main__":
    main()
