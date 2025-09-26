import asyncio
import configparser
import os
import zipfile
from typing import List, Dict, Any

import aioboto3
import boto3
import numpy as np
import polars as pl
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

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

async def upload_file_to_s3(session, file_info: Dict[str, Any], bucket: str, use: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Asynchronously upload a single file to S3.
    
    Args:
        session: aioboto3 session
        file_info: Dictionary containing file path and other metadata
        bucket: S3 bucket name
        use: Usage type (24hour, 1hour, etc.)
        semaphore: Semaphore to limit concurrent uploads
    
    Returns:
        Dictionary with upload status
    """
    async with semaphore:  # Limit concurrent uploads
        file_name = file_info['video_path']
        object_name = f'for_maria/{use}_us_videos/{os.path.basename(file_name)}'
        
        try:
            async with session.client('s3') as s3_client:
                with open(file_name, 'rb') as f:
                    await s3_client.upload_fileobj(f, bucket, object_name)
            return {'status': 'success', 'file': file_name}
        except Exception as e:
            return {'status': 'error', 'file': file_name, 'error': str(e)}

async def upload_files_concurrently(video_df: pl.DataFrame, bucket: str, use: str, max_concurrent: int = 10):
    """
    Upload multiple files to S3 concurrently.
    
    Args:
        video_df: Polars DataFrame containing video information
        bucket: S3 bucket name
        use: Usage type (24hour, 1hour, etc.)
        max_concurrent: Maximum number of concurrent uploads
    """
    # Create a semaphore to limit concurrent uploads
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aioboto3 session
    session = aioboto3.Session()
    
    # Create upload tasks
    posts = video_df.to_dicts()
    tasks = []
    
    for post in posts:
        task = upload_file_to_s3(session, post, bucket, use, semaphore)
        tasks.append(task)
    
    # Execute all tasks with progress bar
    results = []
    with tqdm(total=len(tasks), desc=f'Uploading {use} videos to S3') as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
            
            # Log errors if any
            if result['status'] == 'error':
                tqdm.write(f"Error uploading {result['file']}: {result['error']}")
    
    # Summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nUpload complete for {use}:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    return results

async def main_async():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    bytes_dir_paths = config['paths']['mp4_paths'].split(',')

    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # S3 configuration
    bucket = 'tiktok-share-eu-1'
    max_concurrent_uploads = 20  # Adjust based on your network and system capabilities
    
    for use in ['24hour', '1hour']:
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
        for result_path in tqdm(result_paths):
            batch_df = pl.read_parquet(result_path)\
                .filter(pl.col('locationCreated') == 'US')\
                .filter(pl.col('video').struct.field('duration') > 0)\
                .select(['id'])
            if video_df is not None:
                video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
            else:
                video_df = batch_df

        print(f"Found {len(video_df)} videos for {use}")

        video_df = video_df.with_columns([
            pl.col('id').cast(pl.UInt64)
                .map_elements(lambda i: format(i, '064b'), pl.String)
                .str.slice(0, 32)
                .map_elements(lambda s: int(s, 2), pl.UInt64)
                .alias('timestamp'),
            pl.lit(bytes_dir_paths).alias('bytes_dir_paths'),
        ])
        video_df = video_df.explode('bytes_dir_paths').rename({'bytes_dir_paths': 'bytes_dir_path'})
        video_df = video_df.with_columns(
            pl.concat_str([
                pl.col('bytes_dir_path'),
                pl.col('timestamp').cast(pl.String),
                pl.lit('/'),
                pl.col('id').cast(pl.String),
                pl.lit('.mp4'),
            ]).alias('video_path')
        )
        video_df = video_df.with_columns(pl.col('video_path').map_elements(os.path.exists, return_dtype=pl.Boolean, strategy='threading').alias('video_path_exists'))

        # get videos that have a video path that exists
        video_df = video_df.filter(pl.col('video_path_exists'))
        df_path = f"./data/{use}_us_videos.parquet.zstd"
        video_df.write_parquet(df_path, compression='zstd')

        s3_client = boto3.client('s3')

        bucket_name = 'tiktok-share-eu-1'
        s3_object_key = f'for_maria/{use}_us_videos/{os.path.basename(df_path)}' # The name the file will have in S3

        try:
            s3_client.upload_file(df_path, bucket_name, s3_object_key)
            print(f"File '{df_path}' uploaded to S3 bucket '{bucket_name}' as '{s3_object_key}'")
        except Exception as e:
            print(f"Error uploading file: {e}")

        # Upload to S3 concurrently
        await upload_files_concurrently(video_df, bucket, use, max_concurrent_uploads)

def main():
    """Wrapper function to run the async main function."""
    asyncio.run(main_async())

if __name__ == '__main__':
    main()