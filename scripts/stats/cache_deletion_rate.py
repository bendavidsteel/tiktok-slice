import json
import os

import polars as pl
import tqdm

def extract_video_data(df):
    return df.filter(
        pl.col('return').struct.field('id').is_not_null()
    ).with_columns([
        pl.col('return').struct.field('id').alias('video_id'),
        pl.col('return').struct.field('createTime').cast(pl.UInt64).alias('createTime'),
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
    ]).select(['video_id', 'desc', 'createTime', 'authorUniqueId', 'commentCount', 'diggCount', 'shareCount', 'playCount', 'videoDuration', 'isImagePost', 'numImages', 'locationCreated'])


def main():
    video_df = None
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10')
    video_pbar = tqdm.tqdm(total=60*60 + 24*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'results.parquet.gzip':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_result_df = pl.read_parquet(result_path, columns=['result'])
                batch_result_df = batch_result_df.filter(
                    pl.col('result').is_not_null() & 
                    pl.col('result').struct.field('return').is_not_null()
                ).with_columns(
                    pl.col('result').struct.field('return').alias('return')
                )
                
                error_df = batch_result_df.with_columns(
                    pl.col('return').struct.field('statusMsg').fill_null('success').alias('statusMsg'),
                    pl.col('result').struct.field('post_time')
                ).select(['statusMsg', 'post_time'])
                error_df.write_parquet(os.path.join(root, 'errors.parquet.zstd'), compression='zstd')


if __name__ == '__main__':
    main()