import json
import os

import polars as pl
import tqdm

def extract_video_data(df):
    video_df = df.filter(pl.col('return').struct.field('id').is_not_null()).select(pl.col('return').struct.unnest())
    
    video_df = video_df.with_columns([
        pl.col('id').alias('video_id'),
        pl.col('createTime').cast(pl.UInt64),
        pl.col('desc'),
        pl.col('author').struct.field('uniqueId').alias('authorUniqueId'),
        pl.col('author').struct.field('verified').alias('authorVerified'),
        pl.col('author').struct.field('nickname').alias('authorNickname'),
        pl.col('author').struct.field('signature').alias('authorSignature'),
        pl.col('music').struct.field('original').alias('musicOriginal'),
        pl.col('music').struct.field('title').alias('musicTitle'),
        pl.col('stats').struct.field('commentCount').cast(pl.Int64).alias('commentCount'),
        pl.col('stats').struct.field('diggCount').cast(pl.Int64).alias('diggCount'),
        pl.col('stats').struct.field('shareCount').cast(pl.Int64).alias('shareCount'),
        pl.col('stats').struct.field('playCount').cast(pl.Int64).alias('playCount'),
        pl.col('video').struct.field('duration').cast(pl.Int64).alias('videoDuration'),
        pl.col('video').struct.field('videoQuality').alias('videoQuality'),
        pl.col('imagePost').is_not_null().alias('isImagePost'),
        pl.col('locationCreated'),
        pl.col('diversificationLabels')
    ])

    if 'imagePost' in video_df.columns:
        video_df = video_df.with_columns(pl.col('imagePost').struct.field('images').list.len().alias('numImages'))

    if 'VQScore' in dict(video_df.schema['video']):
        video_df = video_df.with_columns(pl.col('video').struct.field('VQScore').alias('videoVQScore'))

    return video_df


def main():
    result_df = None
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10')
    video_pbar = tqdm.tqdm(total=60*60 + 24*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'results.parquet.gzip':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_result_df = pl.read_parquet(result_path, columns=['result', 'args'])
                batch_result_df = batch_result_df.with_columns([
                    pl.col('args').cast(pl.UInt64),
                        ((pl.col('result').struct.field('return').struct.field('statusMsg') != "item doesn't exist")\
                        | pl.col('result').struct.field('return').struct.field('id').is_not_null()).alias('success')
                    ])\
                    .select(['args', 'success'])
                
                if result_df is not None:
                    result_df = pl.concat([result_df, batch_result_df])
                else:
                    result_df = batch_result_df

    result_df.write_parquet('./data/stats/all/hit_rate.parquet.zstd', compression='zstd')


if __name__ == '__main__':
    main()