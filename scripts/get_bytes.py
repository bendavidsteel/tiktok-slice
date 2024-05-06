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

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    result_paths = []
    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.json':
                result_paths.append(os.path.join(dir_path, filename))
    # for result_path in tqdm.tqdm(result_paths, desc="Reading result files"):
    
    all_results = process_amap(read_result_path, result_paths, num_workers=multiprocessing.cpu_count() - 1, pbar_desc="Reading result files")
    all_results = [v for res in all_results for v in res]
    video_results = [r for r in all_results if r['result']['return'] and 'statusMsg' not in r['result']['return']]
    videos = [r['result']['return'] for r in video_results]

    bytes_dir_path = os.path.join(data_dir_path, "results", 'bytes')
    if not os.path.exists(bytes_dir_path):
        os.makedirs(bytes_dir_path)
    else:
        fetched_filenames = os.listdir(bytes_dir_path)
        fetched_ids = [int(f.split('.')[0]) for f in fetched_filenames]

    method = 'pytok'
    if method == 'tiktokapi':
        async with TikTokApi() as api:
            await api.create_sessions(ms_tokens=[None], num_sessions=1, sleep_after=3)
            for video_data in videos:
                video = api.video(
                    url=f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_data['id']}",
                    data=video_data
                )
                video_info = await video.info()
                bytes = await video.bytes()
                pass
    elif method == 'pytok':
        for video_data in tqdm.tqdm(videos):
            if video_data['id'] in fetched_ids:
                continue
            try:
                async with PyTok(headless=True) as api:
                    video = api.video(
                        url=f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_data['id']}",
                        # data=video_data
                    )
                    video_info = await video.info()
                    bytes = await video.bytes()
                    with open(os.path.join(bytes_dir_path, f"{video_data['id']}.mp4"), 'wb') as f:
                        f.write(bytes)
            except pytok.exceptions.NotAvailableException:
                continue
            except pytok.exceptions.TimeoutException:
                continue
    elif method == 'requests':
        for video_data in tqdm.tqdm(videos):
            if video_data['video']['downloadAddr']:
                video_id = video_data['id']
                url = f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_id}"
                headers = get_headers()
                
                info_res = requests.get(url, headers=headers)
                video_processor = ProcessVideo(info_res)
                text_chunk = info_res.text
                do = video_processor.process_chunk(text_chunk)

                video_d = video_processor.process_response()
                cookies = {c.name: c.value for c in info_res.cookies}
                cookies['bm_sv'] = """41CDD82A3A77D5CCE2B54956EFFBD484~2v+fgMc/HS3XXYBU8DwOrq6CfX2ufpT4w9woRIdO0nl7zCUij0wstoi3DYubs+YquYCQ7WQxx+a5iXEYmA6bTOHqjgAwHDgmdYy+td8sYgU8wqCgjNG0oHmvnWE3JyaePt37uro2bNpZeWSUKFoaYvBFrYs3y1EBXG5nLn9g="""
                bytes_res = requests.get(video_d['video']['downloadAddr'], headers=headers, cookies=cookies)
                if 200 <= bytes_res.status_code < 300:
                    with open(os.path.join(bytes_dir_path, f"{video_data['id']}.mp4"), 'wb') as f:
                        f.write(bytes_res.content)


if __name__ == "__main__":
    asyncio.run(main())