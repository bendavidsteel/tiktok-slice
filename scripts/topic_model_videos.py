import os

from bertopic import BERTopic
import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    bytes_dir_path = os.path.join(this_dir_path, '..', 'data', 'bytes')
    embeddings = None
    video_df = None
    for dir_name in os.listdir(bytes_dir_path):
        filenames = os.listdir(os.path.join(bytes_dir_path, dir_name))
        if 'video_embeddings.npy' in filenames and 'videos.parquet.gzip' in filenames:
            batch_embeddings = np.load(os.path.join(bytes_dir_path, dir_name, 'video_embeddings.npy'), allow_pickle=True)
            if not batch_embeddings.shape:
                continue

            batch_video_df = pd.read_parquet(os.path.join(bytes_dir_path, dir_name, 'videos.parquet.gzip'))
            if embeddings is None:
                embeddings = batch_embeddings
            else:
                embeddings = np.concatenate([embeddings, batch_embeddings])

            if video_df is None:
                video_df = batch_video_df
            else:
                video_df = pd.concat([video_df, batch_video_df])

    # Train our model with images only
    topic_model = BERTopic(min_topic_size=30)
    topics, probs = topic_model.fit_transform(documents=video_df['video'].map(lambda v: v['desc']), embeddings=embeddings)

    df = topic_model.get_topic_info()
    pass

if __name__ == '__main__':
    main()