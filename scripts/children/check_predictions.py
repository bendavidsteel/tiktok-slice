import configparser
import os
import zipfile

import polars as pl


def get_existing_paths(df, bytes_dir_paths):
    df = df.with_columns(pl.col('id').map_elements(lambda i: int(format(int(i), '064b')[:32], 2), pl.UInt64).alias('timestamp'))\
        .with_columns(pl.lit(bytes_dir_paths).alias('bytes_dir_paths'))\
        .explode('bytes_dir_paths')\
        .rename({'bytes_dir_paths': 'bytes_dir_path'})\
        .with_columns(pl.concat_str([
            pl.col('bytes_dir_path'),
            pl.col('timestamp').cast(pl.String),
            pl.lit('/'),
            pl.col('id').cast(pl.String),
            pl.lit('.mp4')
        ]).alias('file_path'))\
        .with_columns(pl.col('file_path').map_elements(lambda p: os.path.exists(p), pl.Boolean).alias('exists'))
    return df.filter(pl.col('exists'))

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    bytes_dir_paths = config['paths']['mp4_paths'].split(',')

    video_df = pl.read_parquet(os.path.join('.', 'data', 'stats', '24hour', 'video_child_prob.parquet.gzip'))
    
    # Process video data
    threshold = 0.43
    video_df = video_df.with_columns((pl.col('child_prob') > threshold).cast(pl.Int32).alias('child_present'))

    # get file paths to examples
    # get classified child present
    child_df = get_existing_paths(video_df.filter(pl.col('child_present') == 1).sample(100), bytes_dir_paths).sample(20)

    output_zip_path = os.path.join('.', 'data', 'child_video_examples.zip')
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        # Add child present videos
        for post in child_df.select(['id', 'file_path']).to_dicts():
            arcname = os.path.join('child_present', f"{post['id']}.mp4")
            zipf.write(post['file_path'], arcname=arcname)

        # Add non-child videos
        non_child_df = get_existing_paths(video_df.filter(pl.col('child_present') == 0).sample(100), bytes_dir_paths).sample(20)
        for post in non_child_df.select(['id', 'file_path']).to_dicts():
            arcname = os.path.join('non_child', f"{post['id']}.mp4")
            zipf.write(post['file_path'], arcname=arcname)

        # add some random samples
        sample_df = get_existing_paths(video_df.sample(200), bytes_dir_paths).sample(100)
        for post in sample_df.select(['id', 'file_path']).to_dicts():
            arcname = os.path.join('random_samples', f"{post['id']}.mp4")
            zipf.write(post['file_path'], arcname=arcname)

if __name__ == '__main__':
    main()