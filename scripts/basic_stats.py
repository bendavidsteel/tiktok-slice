import os

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from convert_json_to_parquet import get_result_paths


def get_video_df(result_path):
    result_df = pd.read_parquet(result_path, columns=['result', 'args'])
    result_df['return'] = result_df['result'].map(lambda r: r['return'])
    result_df['id'] = result_df['return'].map(lambda r: r['id'] if r and 'id' in r else None)
    video_df = result_df[result_df['id'].map(lambda i: i is not None)]
    
    return video_df

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    result_paths = list(get_result_paths(data_dir_path))
    result_paths = sorted(result_paths)

    df = None
    result_paths = result_paths[:10]
    for result_path in tqdm.tqdm(result_paths):
        batch_df = get_video_df(result_path)

        if df is None:
            df = batch_df
        else:
            df = pd.concat([df, batch_df])

    df['desc'] = df['return'].map(lambda r: r['desc'])
    df['authorUniqueId'] = df['return'].map(lambda r: r['author']['uniqueId'] if r['author'] else None)
    df['commentCount'] = df['return'].map(lambda r: int(r['stats']['commentCount']) if r['stats'] else None)
    df['diggCount'] = df['return'].map(lambda r: int(r['stats']['diggCount']) if r['stats'] else None)
    df['shareCount'] = df['return'].map(lambda r: int(r['stats']['shareCount']) if r['stats'] else None)
    df['playCount'] = df['return'].map(lambda r: int(r['stats']['playCount']) if r['stats'] else None)
    df['videoDuration'] = df['return'].map(lambda r: r['video']['duration'])
    df['isImagePost'] = df['return'].map(lambda r: 'imagePost' in r)
    df['numImages'] = df['return'].map(lambda r: len(r['imagePost']['images']) if 'imagePost' in r and r['imagePost'] else 0)
    df['locationCreated'] = df['return'].map(lambda r: r['locationCreated'])
    
    print(f"Number of unique users: {len(df['authorUniqueId'].unique())}")
    print(f"Average number of comments: {df['commentCount'].mean()}")
    print(f"Average number of likes: {df['diggCount'].mean()}")
    print(f"Average number of shares: {df['shareCount'].mean()}")
    print(f"Average number of plays: {df['playCount'].mean()}")
    print(f"Average video duration: {df['videoDuration'].mean()}")

    fig, axes = plt.subplots(nrows=1, ncols=5)
    axes[0].hist(df['commentCount'], bins=50)
    axes[0].set_x


if __name__ == "__main__":
    main()