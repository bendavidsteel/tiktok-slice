import datetime
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl
import tqdm

def main():
    comments_dir_path = os.path.join('.', 'data', 'comments')
    comment_df = None
    comment_pbar = tqdm.tqdm(total=len(list(os.listdir(comments_dir_path))), desc='Reading comments')
    for dir_name in os.listdir(comments_dir_path):
        comment_pbar.update(1)
        if dir_name.endswith('.zip'):
            continue
        comment_path = os.path.join(comments_dir_path, dir_name, 'comments.parquet.zstd')
        if not os.path.exists(comment_path):
            continue
        file_df = pl.read_parquet(comment_path)
        if comment_df is None:
            comment_df = file_df
        else:
            comment_df = pl.concat([comment_df, file_df], how='diagonal_relaxed')

    comment_df = comment_df.with_columns(pl.from_epoch('create_time').alias('create_time'))

    second_count_df = comment_df.with_columns(pl.col('create_time').dt.second().alias('second')).group_by('second').count().sort('second')
    minute_count_df = comment_df.with_columns(pl.col('create_time').dt.minute().alias('minute')).group_by('minute').count().sort('minute')

    # calculcate std dev
    print(f"Second std dev: {second_count_df['count'].std()}")
    print(f"Minute std dev: {minute_count_df['count'].std()}")
    print(f"Second mean: {second_count_df['count'].mean()}")
    print(f"Minute mean: {minute_count_df['count'].mean()}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].bar(second_count_df['second'], second_count_df['count'])
    axes[0].set_xlabel("Second")
    axes[0].set_ylabel("Number of Comments")
    axes[1].bar(minute_count_df['minute'], minute_count_df['count'])
    axes[1].set_xlabel("Minute")
    axes[1].set_ylabel("Number of Comments")
    fig.savefig('./figs/comments_per_time_interval.png')

    # look at average time between comments
    user_comments_df = comment_df.group_by('uid').agg(pl.col('create_time').sort().diff().mean().alias('avg_time_between_comments'), pl.col('create_time').len().alias('num_comments'))\
        .filter(pl.col('num_comments') > 1)\
        .with_columns(pl.col('avg_time_between_comments').dt.total_seconds().alias('avg_time_between_comments'))
    # plot histogram of average time between comments
    fig, ax = plt.subplots()
    ax.hist(user_comments_df['avg_time_between_comments'], bins=100)
    ax.set_xlabel("Average Time Between Comments")
    ax.set_ylabel("Number of Users")
    fig.savefig('./figs/avg_time_between_comments.png')

    suspicious_comments_df = comment_df.join(user_comments_df.filter((pl.col('avg_time_between_comments') < 3) & (pl.col('num_comments') > 10)), on='uid')
    
    emoji_regex = "[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+$"
    
    suspicious_comments_df = suspicious_comments_df.with_columns(pl.col('text').str.contains(emoji_regex).alias('contains_emoji'))
    pass

if __name__ == '__main__':
    main()
