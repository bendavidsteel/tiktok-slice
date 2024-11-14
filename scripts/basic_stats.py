import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tqdm

from result_funcs import get_result_paths

def get_video_df(result_df):
    return result_df.filter(
        pl.col('return').struct.field('id').is_not_null()
    ).with_columns(
        pl.col('return').struct.field('id').alias('id')
    )

def extract_video_data(df):
    return df.with_columns([
        pl.col('return').struct.field('id').alias('id'),
        pl.col('return').struct.field('author').struct.field('uniqueId').alias('authorUniqueId'),
        pl.col('return').struct.field('stats').struct.field('commentCount').cast(pl.Int64).alias('commentCount'),
        pl.col('return').struct.field('stats').struct.field('diggCount').cast(pl.Int64).alias('diggCount'),
        pl.col('return').struct.field('stats').struct.field('shareCount').cast(pl.Int64).alias('shareCount'),
        pl.col('return').struct.field('stats').struct.field('playCount').cast(pl.Int64).alias('playCount'),
        pl.col('return').struct.field('video').struct.field('duration').cast(pl.Float64).alias('videoDuration'),
        pl.col('return').struct.field('locationCreated').alias('locationCreated')
    ]).select(['id', 'authorUniqueId', 'commentCount', 'diggCount', 'shareCount', 'playCount', 'videoDuration', 'locationCreated'])

def extract_error_data(df):
    return df.filter(
        pl.col('return').struct.field('statusCode').is_not_null()
    ).with_columns([
        pl.col('return').struct.field('statusCode').cast(pl.Int32).alias('statusCode'),
        pl.col('return').struct.field('statusMsg').alias('statusMsg')
    ]).select(['statusCode', 'statusMsg'])

def plot_loglog_hist(x, title, xlabel, ylabel, fig_path, bins=50):
    fig, ax = plt.subplots()
    x = x.drop_nulls()
    hist, bins = np.histogram(x.to_numpy(), bins=bins)
    if bins[0] == 0:
        start = 1
        logbins = np.logspace(np.log10(start),np.log10(bins[-1]),len(bins)-1)
        logbins = np.concatenate([np.array([0]), logbins])
    else:
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.hist(x.to_numpy(), bins=logbins)
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(fig_path)

def get_video_stats(result_paths, output_dir_path):

    video_df = None
    result_paths = result_paths[:5]
    for result_path in tqdm.tqdm(result_paths):
        batch_result_df = pl.read_parquet(result_path, columns=['result', 'args'])
        batch_result_df = batch_result_df.filter(
            pl.col('result').is_not_null() & 
            pl.col('result').struct.field('return').is_not_null()
        ).with_columns(
            pl.col('result').struct.field('return').alias('return')
        ).drop('result')
        batch_df = get_video_df(batch_result_df)
        batch_df = extract_video_data(batch_df)

        video_df = pl.concat([video_df, batch_df]) if video_df is not None else batch_df

    print(f"Number of videos: {video_df.height}")
    print(f"Number of unique users: {video_df['authorUniqueId'].n_unique()}")
    print(f"Average number of comments: {video_df['commentCount'].mean()}")
    print(f"Average number of likes: {video_df['diggCount'].mean()}")
    print(f"Average number of shares: {video_df['shareCount'].mean()}")
    print(f"Average number of plays: {video_df['playCount'].mean()}")
    print(f"Average video duration: {video_df['videoDuration'].mean()}")

    video_df.group_by('authorUniqueId').count().sort('count', descending=True).write_csv(os.path.join(output_dir_path, "author_unique_id_value_counts.csv"))
    video_df.group_by('locationCreated').count().sort('count', descending=True).write_csv(os.path.join(output_dir_path, "location_created_value_counts.csv"))
    video_df.select('id').unique().write_csv(os.path.join(output_dir_path, "video_ids.csv"))

    plot_loglog_hist(video_df['commentCount'], "Comment Count Histogram", "Comment Count", "Frequency", os.path.join(output_dir_path, "comment_count_hist.png"))
    plot_loglog_hist(video_df['diggCount'], "Like Count Histogram", "Like Count", "Frequency", os.path.join(output_dir_path, "like_count_hist.png"))
    plot_loglog_hist(video_df['shareCount'], "Share Count Histogram", "Share Count", "Frequency", os.path.join(output_dir_path, "share_count_hist.png"))
    plot_loglog_hist(video_df['playCount'], "Play Count Histogram", "Play Count", "Frequency", os.path.join(output_dir_path, "play_count_hist.png"))
    plot_loglog_hist(video_df['videoDuration'], "Video Duration Histogram", "Video Duration", "Frequency", os.path.join(output_dir_path, "video_duration_hist.png"))

def get_error_stats(result_paths, output_dir_path):
    error_df = None
    # result_paths = result_paths[198:203]
    for result_path in tqdm.tqdm(result_paths):
        batch_result_df = pl.read_parquet(result_path, columns=['result', 'args'])
        batch_result_df = batch_result_df.filter(
            pl.col('result').is_not_null() & 
            pl.col('result').struct.field('return').is_not_null()
        ).with_columns(
            pl.col('result').struct.field('return').alias('return')
        ).drop('result')
        batch_error_df = extract_error_data(batch_result_df)

        error_df = pl.concat([error_df, batch_error_df]) if error_df is not None else batch_error_df

    # get error value counts
    error_df.group_by(['statusCode', 'statusMsg']).count().sort('count', descending=True).write_csv(os.path.join(output_dir_path, "error_value_counts.csv"))


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data", "results", "2024_04_10", "hours", "19")
    output_dir_path = os.path.join(this_dir_path, "..", "data", "results", "2024_04_10", "outputs")
    result_paths = list(get_result_paths(data_dir_path))
    result_paths = sorted(result_paths)

    get_video_stats(result_paths, output_dir_path)
    get_error_stats(result_paths, output_dir_path)

if __name__ == "__main__":
    main()