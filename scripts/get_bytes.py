import asyncio
import json
import multiprocessing
import os

from pytok.tiktok import PyTok
from TikTokApi import TikTokApi

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
    
    all_results = process_amap(read_result_path, result_paths[:3], num_workers=multiprocessing.cpu_count() - 1, pbar_desc="Reading result files")
    all_results = [v for res in all_results for v in res]
    video_results = [r for r in all_results if r['result']['return'] and 'statusMsg' not in r['result']['return']]
    videos = [r['result']['return'] for r in video_results]

    bytes_dir_path = os.path.join(data_dir_path, "results", 'bytes')
    if not os.path.exists(bytes_dir_path):
        os.makedirs(bytes_dir_path)

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
        for video_data in videos:
            async with PyTok() as api:
                video = api.video(
                    url=f"https://www.tiktok.com/@{video_data['author']['uniqueId']}/video/{video_data['id']}",
                    # data=video_data
                )
                video_info = await video.info()
                bytes = await video.bytes()
                with open(os.path.join(bytes_dir_path, f"{video_data['id']}.mp4"), 'wb') as f:
                    f.write(bytes)


if __name__ == "__main__":
    asyncio.run(main())