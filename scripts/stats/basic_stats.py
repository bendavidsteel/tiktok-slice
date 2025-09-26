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

def process_batch(result_path, val_count_dfs):
    # batch_result_df = pl.read_parquet(result_path, columns=['result', 'args'])
    # batch_result_df = batch_result_df.filter(
    #     pl.col('result').is_not_null() & 
    #     pl.col('result').struct.field('return').is_not_null()
    # ).with_columns(
    #     pl.col('result').struct.field('return').alias('return')
    # ).drop('result')
    
    video_df = pl.read_parquet(result_path)
    # error_df = extract_error_data(batch_result_df)

    # Update value counts
    if val_count_dfs:
        author_counts_df, location_counts_df, error_counts_df, comment_counts_df, like_counts_df, share_counts_df, play_counts_df, duration_counts_df, follower_counts_df = val_count_dfs
    else:
        author_counts_df, location_counts_df, error_counts_df, comment_counts_df, like_counts_df, share_counts_df, play_counts_df, duration_counts_df, follower_counts_df = None, None, None, None, None, None, None, None, None
    author_counts_df = update_value_counts(video_df, 'authorUniqueId', author_counts_df)
    location_counts_df = update_value_counts(video_df, 'locationCreated', location_counts_df)
    # error_counts_df = update_value_counts(error_df, ['statusCode', 'statusMsg'], error_counts_df)
    
    # Update distribution counts
    comment_counts_df = update_value_counts(video_df, 'commentCount', comment_counts_df)
    like_counts_df = update_value_counts(video_df, 'diggCount', like_counts_df)
    share_counts_df = update_value_counts(video_df, 'shareCount', share_counts_df)
    play_counts_df = update_value_counts(video_df, 'playCount', play_counts_df)
    duration_counts_df = update_value_counts(video_df, 'videoDuration', duration_counts_df)
    follower_counts_df = update_value_counts(video_df, pl.col()'followerCount', follower_counts_df)
    
    # Aggregate statistics
    batch_df = video_df.select(['video_id', 'commentCount'])
    
    val_count_dfs = author_counts_df, location_counts_df, error_counts_df, comment_counts_df, like_counts_df, share_counts_df, play_counts_df, duration_counts_df, follower_counts_df

    return batch_df, val_count_dfs

def finalize_results(unique_video_df, output_dir_path, val_count_dfs):

    # Calculate and print overall statistics
    total_videos = len(unique_video_df)
    print(f"Number of videos: {total_videos}")

    author_counts_df, location_counts_df, error_counts_df, comment_counts_df, like_counts_df, share_counts_df, play_counts_df, duration_counts_df, follower_counts_df = val_count_dfs

    # Convert final parquet files to CSV
    author_counts_df.write_csv(os.path.join(output_dir_path, "author_unique_id_value_counts.csv"))
    print(f"Number of unique users: {author_counts_df['authorUniqueId'].n_unique()}")
    user_df = None
    location_counts_df.write_csv(os.path.join(output_dir_path, "location_created_value_counts.csv"))
    # error_counts_df.write_csv(os.path.join(output_dir_path, "error_value_counts.csv"))

    comment_counts_df.write_csv(os.path.join(output_dir_path, "comment_count_value_counts.csv"))
    like_counts_df.write_csv(os.path.join(output_dir_path, "like_count_value_counts.csv"))
    share_counts_df.write_csv(os.path.join(output_dir_path, "share_count_value_counts.csv"))
    play_counts_df.write_csv(os.path.join(output_dir_path, "play_count_value_counts.csv"))
    duration_counts_df.write_csv(os.path.join(output_dir_path, "video_duration_value_counts.csv"))
    follower_counts_df.write_csv(os.path.join(output_dir_path, "follower_count_value_counts.csv"))
    
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
        batch_df, val_count_dfs = process_batch(result_path, val_count_dfs)
        if video_df is not None:
            video_df = pl.concat([video_df, batch_df])
        else:
            video_df = batch_df

    video_df.write_csv(os.path.join(output_dir_path, "video_ids.csv"))

    # Finalize results
    finalize_results(video_df, output_dir_path, val_count_dfs)

if __name__ == "__main__":
    main()
