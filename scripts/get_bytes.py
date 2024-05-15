import asyncio
import json
import multiprocessing
import os

import requests
import tqdm

from pytok.tiktok import PyTok
import pytok.exceptions
from TikTokApi import TikTokApi

from get_random_sample import get_headers, ProcessVideo
from map_funcs import process_amap

def read_result_path(result_path):
    with open(result_path, 'r') as f:
        try:
            results = json.load(f)
        except:
            return []
    return [r for r in results]

def get_ids_to_get_bytes(data_dir_path, bytes_dir_path):
    fetched_filenames = os.listdir(bytes_dir_path)
    fetched_ids = [int(f.split('.')[0]) for f in fetched_filenames]

    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.json':
                result_path = os.path.join(dir_path, filename)
                results = read_result_path(result_path)
    
                video_results = [r for r in results if r['result']['return'] and 'statusMsg' not in r['result']['return']]
                videos = [r['result']['return'] for r in video_results]

                for video in videos:
                    if int(video['id']) in fetched_ids:
                        continue
                    yield video

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    bytes_dir_path = os.path.join('/', 'media', 'elements_harddrive', 'repos', 'what-for-where', 'data', 'bytes')
    if not os.path.exists(bytes_dir_path):
        os.makedirs(bytes_dir_path)
    videos = get_ids_to_get_bytes(data_dir_path, bytes_dir_path)

    for video_data in tqdm.tqdm(videos):
        if 'video' in video_data and 'downloadAddr' in video_data['video'] and video_data['video']['downloadAddr']:
            try:
                video_id = video_data['id']
                url = f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_id}"
                headers = get_headers()
                
                info_res = requests.get(url, headers=headers)
                video_processor = ProcessVideo(info_res)
                text_chunk = info_res.text
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

                if 'video' not in video_d:
                    continue

                cookies = {c.name: c.value for c in info_res.cookies}
                bytes_res = requests.get(video_d['video']['downloadAddr'], headers=bytes_headers, cookies=cookies)
                if 200 <= bytes_res.status_code < 300:
                    with open(os.path.join(bytes_dir_path, f"{video_data['id']}.mp4"), 'wb') as f:
                        f.write(bytes_res.content)
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    asyncio.run(main())