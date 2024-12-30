import asyncio
import datetime
import json
import os
import subprocess

import dotenv
import httpx
import pandas as pd
import tqdm

from get_random_sample import get_headers, ProcessVideo
from map_funcs import async_amap

def read_result_path(result_path):
    with open(result_path, 'r') as f:
        try:
            results = json.load(f)
        except:
            return []
    return [r for r in results]

def get_ids_to_get_bytes(data_dir_path, read_bytes_dir_paths):
    fetched_ids = []
    for bytes_dir_path in read_bytes_dir_paths:
        for dir_path, dir_names, filenames in os.walk(bytes_dir_path):
            fetched_ids += [int(f.split('.')[0]) for f in filenames if f.endswith('mp4')]
    fetched_ids = set(fetched_ids)

    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.parquet.gzip':
                result_path = os.path.join(dir_path, filename)
                result_df = pd.read_parquet(result_path)
                result_df['return'] = result_df['result'].map(lambda r: r['return'])
                result_df = result_df[result_df['return'].map(lambda r: r is not None)]
                result_df = result_df[result_df['return'].map(lambda r: 'id' in r and r['id'] is not None)]
                result_df.loc[:, 'id'] = result_df['return'].map(lambda r: r['id'])
                result_df.loc[:, 'id'] = result_df['id'].map(int)
                result_df = result_df[~result_df['id'].isin(fetched_ids)]
                videos = result_df['return'].tolist()
    
                yield videos

async def fetch_video_data(video_data):
    if 'video' in video_data and 'downloadAddr' in video_data['video'] and video_data['video']['downloadAddr']:
        try:
            video_id = video_data['id']
            url = f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_id}"
            headers = get_headers()

            id_bits = format(int(video_id), '064b')
            timestamp_bits = id_bits[:32]
            timestamp = int(timestamp_bits, 2)
            bytes_dir_path = os.path.join('/', 'media', 'bsteel', 'TT_DAY', 'TikTok_Hour', 'mp4s')
            timestamp_dir = os.path.join(bytes_dir_path, str(timestamp))
            
            async with httpx.AsyncClient() as client:
                info_res = await client.get(url, headers=headers)
                if info_res.status_code != 200:
                    return None, None
                text_chunk = info_res.text
                video_processor = ProcessVideo()
                do = video_processor.process_chunk(text_chunk)

                bytes_headers = {
                    'sec-ch-ua': '"HeadlessChrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"', 
                    'referer': 'https://www.tiktok.com/', 
                    'accept-encoding': 'identity;q=1, *;q=0', 
                    'sec-ch-ua-mobile': '?0', 
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.4 Safari/537.36', 
                    'range': 'bytes=0-', 
                    'sec-ch-ua-platform': '"Windows"'
                }

                video_d = video_processor.process_response()

                timestamp = datetime.datetime.now().timestamp()

                if 'video' not in video_d:
                    return video_d, timestamp

                if not video_d['video']['downloadAddr']:
                    return video_d, timestamp

                cookies = {c: info_res.cookies[c] for c in info_res.cookies}
                bytes_res = await client.get(video_d['video']['downloadAddr'], headers=bytes_headers, cookies=cookies)
                if 200 <= bytes_res.status_code >= 300:
                    return video_d, timestamp
                content = bytes_res.content

                with open(os.path.join(timestamp_dir, f"{video_id}.mp4"), 'wb') as f:
                    f.write(content)
        except Exception as e:
            print(e)
            return None, None
        else:
            return video_d, timestamp
    return None, None

async def worker(queue, bytes_dir_path, pbar):
    while True:
        video_data = await queue.get()
        if video_data is None:
            break
        await fetch_video_data(video_data, bytes_dir_path)
        queue.task_done()
        pbar.update(1)

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    mount_dir_path = os.path.join('/', 'media', 'bsteel')
    read_bytes_dir_paths = [
            os.path.join(mount_dir_path, 'NAS', 'TikTok_Hour', 'mp4s'),
            os.path.join(mount_dir_path, 'TT_DAY', 'TikTok_Hour', 'mp4s')
    ]
    write_bytes_dir_path = os.path.join(mount_dir_path, 'TT_DAY', 'TikTok_Hour', 'mp4s')
    
    for bytes_dir_path in read_bytes_dir_paths + [write_bytes_dir_path]:
        if not os.path.exists(bytes_dir_path):
            os.makedirs(bytes_dir_path)

    dotenv.load_dotenv()

    username = os.environ['USERNAME']
    host = os.environ['HOST']
    server_dir_path = os.environ['SERVER_DIR_PATH']
    host_dir_path = os.environ['HOST_DIR_PATH']

    r = subprocess.run(f'rsync -avz {username}@{host}:{server_dir_path}/* {host_dir_path}/', shell=True, capture_output=True)
        
    if r.returncode != 0:
        print(f"Failed to copy files from {server_dir_path} to {host_dir_path}: {r.stderr}")
    
    num_workers = 4  # Adjust the number of workers as needed

    for videos in get_ids_to_get_bytes(data_dir_path, read_bytes_dir_paths):
        if not videos:
            continue
        try:
            video_id = videos[0]['id']
            id_bits = format(int(video_id), '064b')
            timestamp_bits = id_bits[:32]
            timestamp = int(timestamp_bits, 2)
            timestamp_dir = os.path.join(bytes_dir_path, str(timestamp))
            if not os.path.exists(timestamp_dir):
                os.makedirs(timestamp_dir)

            videos_data = await async_amap(fetch_video_data, videos, num_workers=num_workers, progress_bar=True, pbar_desc='Downloading Bytes')
            videos_data = [(v, t) for v, t in videos_data if v is not None]
            if not videos_data:
                continue
            video_df = pd.DataFrame(videos_data, columns=['video', 'timestamp'])
            df_path = os.path.join(timestamp_dir, f"{timestamp}.parquet.gzip")
            try:
                video_df.to_parquet(df_path, compression='gzip')
            except Exception as e:
                video_df.to_json(df_path.replace('.parquet.gzip', '.json'))
                print(e)
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    asyncio.run(main())
