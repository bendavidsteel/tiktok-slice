import asyncio
import json
import os

import pandas as pd
import tqdm

from pytok.tiktok import PyTok

async def main():
    server_region = "canada"
    origin_region = "germany"
    headless = True
    request_delay = 1
    
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data", origin_region)

    if not os.path.exists(os.path.join(data_dir_path, 'video_ips', server_region)):
        os.makedirs(os.path.join(data_dir_path, 'video_ips', server_region))

    video_data = []

    video_dir_path = os.path.join(data_dir_path, "videos")
    videos = []
    for file_name in os.listdir(video_dir_path):
        if file_name.startswith("all"):
            file_path = os.path.join(video_dir_path, file_name)
            if file_name.endswith(".parquet"):
                video_df = pd.read_parquet(file_path)
                videos.extend(video_df.to_dict(orient='records'))
            elif file_name.endswith(".json"):
                with open(file_path, 'r') as file:
                    video_json = json.load(file)
                video_json = [{'author_name': video['author']['uniqueId'], 'video_id': video['id']} for video in video_json]
                videos.extend(video_json)
    

    for video in tqdm.tqdm(videos):
        try:
            async with PyTok(headless=headless, request_delay=request_delay) as api:
                video_obj = api.video(username=video['author_name'], id=video['video_id'])
                video_info = await video_obj.info()
                await asyncio.sleep(request_delay)
            
                network_data = await video_obj.bytes_network_info()
                video_info['network'] = {}
                video_info['network']['play_addr'] = network_data
                with open(os.path.join(data_dir_path, 'video_ips', server_region, f"{video['video_id']}.json"), 'w') as file:
                    json.dump(video_info, file, indent=4)
        except Exception as e:
            pass

if __name__ == "__main__":
    asyncio.run(main())