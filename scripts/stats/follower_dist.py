import os

import polars as pl
from tqdm import tqdm

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", '..', "data", 'topic_model_videos')

    video_df = pl.DataFrame()
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours', '19')
    video_pbar = tqdm(total=60*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                batch_video_df = batch_video_df.select([
                    pl.col('video_id'),
                    pl.col('authorVerified'),
                    pl.col('musicOriginal'),
                    pl.col('videoDuration'),
                    pl.col('videoQuality'),
                    pl.col('locationCreated'),
                    pl.col('desc'),
                    pl.col('shareCount'),
                    pl.col('diggCount'),
                    pl.col('commentCount'),
                    pl.col('playCount'),
                    pl.col('diversificationLabels')
                ])
                video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')

    pass

if __name__ == '__main__':
    main()