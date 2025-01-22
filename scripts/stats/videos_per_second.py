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
            video_df = pl.concat([video_df, batch_df])
        else:
            video_df = batch_df

    video_df = video_df.with_columns(pl.from_epoch('createTime'))\
        .with_columns(pl.col('createTime').dt.second().alias('createSecond'))

    # plot count per second
    second_counts = video_df.group_by('createSecond')\
        .agg(pl.col('createSecond').count().alias('count'))\
        .sort('createSecond')
    fig, ax = plt.subplots()
    ax.bar(second_counts['createSecond'], second_counts['count'])
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Videos")
    plt.tight_layout()
    fig.savefig(os.path.join('.', 'figs', "videos_per_second.png"))

    # look at change in views and description for videos at 0 second and other seconds
    zero_videos = video_df.filter(pl.col('createSecond') == 0).with_columns(pl.col('playCount').fill_nan(0).fill_null(0))
    avg_zero_views = zero_videos.select(pl.col('playCount')).mean()['playCount'][0]
    non_zero_videos = video_df.filter(pl.col('createSecond') != 0).with_columns(pl.col('playCount').fill_nan(0).fill_null(0))
    avg_non_zero_views = non_zero_videos.select(pl.col('playCount')).mean()['playCount'][0]

    # do t test
    t_stat, p_val = stats.ttest_ind(zero_videos.select(pl.col('playCount')).to_numpy(),
                                    non_zero_videos.select(pl.col('playCount')).to_numpy())
    print(f"Average views for videos at 0 second: {avg_zero_views}")
    print(f"Average views for videos at non 0 second: {avg_non_zero_views}")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_val}")

    # look at change in description
    video_df = video_df.with_columns(pl.col('desc').str.extract_all(r'#\w+').alias('hashtags'))

    # look at relative frequency of hashtags in videos at 0 second and other seconds
    zero_hashtags = video_df.filter(pl.col('createSecond') == 0).explode('hashtags')['hashtags'].value_counts()
    non_zero_hashtags = video_df.filter(pl.col('createSecond') != 0).explode('hashtags')['hashtags'].value_counts().with_columns((pl.col('count') / 59).alias('count'))
    hashtag_df = zero_hashtags.join(non_zero_hashtags, on='hashtags', how='outer').fill_null(0).rename({'count': 'count_at_0', 'count_right': 'count_at_non_0'})

    # find which hashtags have biggest gap in both directions
    hashtag_df = hashtag_df.with_columns((pl.col('count_at_0') - pl.col('count_at_non_0')).alias('count_diff'))
    
    print("Hashtags with biggest gap in videos at 0 second")
    print(hashtag_df.sort('count_diff', descending=True).head(10))
    print("Hashtags with biggest gap in videos at non 0 second")
    print(hashtag_df.sort('count_diff').head(10))



if __name__ == "__main__":
    main()