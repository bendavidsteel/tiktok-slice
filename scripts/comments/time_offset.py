import datetime
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl
import tqdm

def main():
    video_df = None
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours', '19')
    video_pbar = tqdm.tqdm(total=60*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                if video_df is None:
                    video_df = batch_video_df
                else:
                    video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')

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

    video_df = video_df.select(['video_id', 'createTime']).rename({'createTime': 'video_create_time'})
    comment_df = comment_df.rename({'create_time': 'comment_create_time'})
    comment_df = comment_df.join(video_df, left_on='aweme_id', right_on='video_id', how='left')
    comment_df = comment_df.filter(pl.col('video_create_time').is_not_null() & pl.col('comment_create_time').is_not_null())
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('comment_create_time')).alias('comment_create_time_epoch'))
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('video_create_time')).alias('video_create_time_epoch'))
    comment_df = comment_df.with_columns((pl.col('comment_create_time_epoch') - pl.col('video_create_time_epoch')).alias('time_offset'))
    comment_df = comment_df.with_columns(pl.col('time_offset').dt.total_seconds().alias('time_offset_seconds'))

    # Convert timedelta to seconds for binning
    time_seconds = comment_df['time_offset'].dt.total_seconds().to_numpy()

    # Create bins in seconds
    bin_edges_seconds = np.geomspace(1, time_seconds.max(), num=100)

    # Calculate histogram manually
    counts, bins = np.histogram(time_seconds, bins=bin_edges_seconds)

    # Plot bars at the center of each bin
    # Width of each bar should be the difference between bin edges
    bar_width = np.diff(bins)
    bar_centers = bins[:-1] + (bar_width / 2)

        # First get comment counts per user
    user_comment_counts = comment_df.group_by('uid').len().rename({'len': 'count'}).sort('count', descending=True)

    # Define the thresholds we want to plot
    comment_thresholds = [1, 5, 25]

    # Create plot
    fig, ax = plt.subplots()

    # Function to create histogram for a given threshold
    def create_histogram_for_threshold(df, threshold):
        # Get users meeting threshold
        users_above_threshold = user_comment_counts.filter(pl.col('count') >= threshold)['uid']
        
        # Filter comments to only those users
        filtered_comments = df.filter(pl.col('uid').is_in(users_above_threshold))
        
        # Get time seconds for these comments
        time_seconds = filtered_comments['time_offset'].dt.total_seconds().to_numpy()
        
        # Create bins and calculate histogram
        bin_edges_seconds = np.geomspace(1, time_seconds.max(), num=100)
        counts, bins = np.histogram(time_seconds, bins=bin_edges_seconds)
        # normalize to density
        counts = counts / counts.sum()
        
        # Calculate bar centers
        bar_centers = bins[:-1] + (np.diff(bins) / 2)
        
        return bar_centers, counts

    # Plot line for each threshold
    for threshold in comment_thresholds:
        try:
            centers, counts = create_histogram_for_threshold(comment_df, threshold)
            ax.plot(centers, counts, label=f'≥{threshold} comments', alpha=0.7)
        except:
            pass

    ax.set_xscale('log')
    # ax.set_yscale('symlog')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time offset')
    ax.set_ylabel('Frequency')
    # ax.set_title('Time offset between video and comment creation\nby commenter frequency')
    ax.legend()

    # Set major ticks
    major_ticks = [
        1,              # 1 second
        60,             # 1 minute
        3600,           # 1 hour
        86400,          # 1 day
        604800,         # 1 week
        2592000,        # 30 days
        31536000        # 365 days
    ]
    ax.set_xticks(major_ticks)

    # Format with datetime.timedelta
    def timedelta_formatter(x, pos):
        seconds = int(x)
        if seconds == 1:
            return "1 second"
        elif seconds == 60:
            return "1 minute"
        elif seconds == 3600:
            return "1 hour"
        elif seconds == 86400:
            return "1 day"
        elif seconds == 604800:
            return "1 week"
        elif seconds == 2592000:
            return "30 days"
        elif seconds == 31536000:
            return "1 year"
        else:
            raise ValueError(f"Unexpected time scale: {seconds}")

    ax.xaxis.set_major_formatter(FuncFormatter(timedelta_formatter))
    plt.xticks(rotation=45)

    fig.savefig('./figs/time_offset_hist_by_frequency_symlog.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()