import asyncio
import datetime
import json
import os

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from pytok.tiktok import PyTok
from pytok.exceptions import NotAvailableException

async def main():
    origin_region = "brazil"

    headless = False
    request_delay = 1

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data", origin_region)

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        os.makedirs(os.path.join(data_dir_path, "videos"))


    if os.path.exists(os.path.join(data_dir_path, "users.csv")):
        user_df = pd.read_csv(os.path.join(data_dir_path, "users.csv"))

        users = user_df[user_df['profile'].notna()]['profile'].str.extract('@(.+)$')[0].to_list()

    if os.path.exists(os.path.join(data_dir_path, "videos", f"search_{origin_region}.parquet")):
        video_df = pd.read_parquet(os.path.join(data_dir_path, "videos", f"search_{origin_region}.parquet"))
        users = video_df['author'].apply(lambda x: x['uniqueId']).unique().tolist()

    already_fetched_users = set()
    for filename in os.listdir(os.path.join(data_dir_path, "videos")):
        if filename.startswith("all"):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir_path, "videos", filename), "r") as f:
                    videos = json.load(f)
                if 'author' in videos:
                    # reload as df
                    videos = pd.DataFrame(videos).to_dict(orient="records")
            if filename.endswith(".parquet"):
                videos = pd.read_parquet(os.path.join(data_dir_path, "videos", filename)).to_dict(orient="records")
            already_fetched_users.update([video['author']['uniqueId'] for video in videos])
    users = [user for user in users if user not in already_fetched_users]

    if len(users) > 0:
        videos = []
        for user in tqdm(users[:10]):
            try:
                async with PyTok(headless=headless, request_delay=request_delay) as api:
                    user_obj = api.user(username=user)
                    user_info = await user_obj.info()
                    async for video in user_obj.videos():
                        video_info = await video.info()
                        videos.append(video_info)
            except NotAvailableException:
                raise
            except Exception as e:
                raise
                print(f"Error fetching videos for user {user}: {e}")

        video_df = pd.DataFrame(videos)
        today = datetime.datetime.today()
        video_df.to_parquet(os.path.join(data_dir_path, "videos", f"all_{today.strftime('%d%m%y')}.parquet"))
    else:
        hashtag_names = {
            'brazil': 'brasil',
            'nigeria': 'nigeria',
            'indonesia': 'indonesia',
        }
        hashtag_name = hashtag_names[origin_region]
        videos = []
        async with PyTok(headless=headless, request_delay=request_delay) as api:
            hashtag_obj = api.hashtag(name=hashtag_name)
            async for video in atqdm(hashtag_obj.videos(count=1000), total=1000):
                video_info = await video.info()
                videos.append(video_info)
    
        video_df = pd.DataFrame(videos)
        video_df.to_parquet(os.path.join(data_dir_path, "videos", f"all_{origin_region}.parquet"))

if __name__ == "__main__":
    asyncio.run(main())