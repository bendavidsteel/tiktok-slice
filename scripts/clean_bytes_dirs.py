import datetime
import json
import os
import re

import pandas as pd
import tqdm

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    bytes_dir_path = os.path.join('/', 'media', 'bsteel', 'Elements', 'tiktok', 'mp4s')
    if not os.path.exists(bytes_dir_path):
        os.makedirs(bytes_dir_path)
    
    for timestamp_dirname in tqdm.tqdm(os.listdir(bytes_dir_path)):
        videos = []
        if not os.path.isdir(os.path.join(bytes_dir_path, timestamp_dirname)):
            continue
        for filename in os.listdir(os.path.join(bytes_dir_path, timestamp_dirname)):
            if filename.endswith('.mp4'):
                continue
            elif filename.endswith(".json"):
                with open(os.path.join(bytes_dir_path, timestamp_dirname, filename), 'r') as f:
                    video_data = json.load(f)
                timestamp = re.search(r'\d{10}\.\d{0,6}', filename).group(0)
                videos.append((video_data, timestamp))
            elif filename.endswith('.parquet.gzip'):
                continue
            else:
                raise ValueError(f"Unexpected file: {filename}")

        if videos:
            video_df = pd.DataFrame(videos, columns=['video', 'timestamp'])
            timestamp = int(datetime.datetime.now().timestamp())
            video_df.to_parquet(os.path.join(bytes_dir_path, timestamp_dirname, f'videos_{timestamp}.parquet.gzip'), compression='gzip')
            for filename in os.listdir(os.path.join(bytes_dir_path, timestamp_dirname)):
                if filename.endswith('.json'):
                    os.remove(os.path.join(bytes_dir_path, timestamp_dirname, filename))


if __name__ == "__main__":
    main()