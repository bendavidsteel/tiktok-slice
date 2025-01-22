import configparser
import dotenv
import joblib
import os

import bertopic
import numpy as np
import polars as pl
from tqdm import tqdm
import umap

from topic_model_videos import get_videos_embeddings, ExtendedTopicModel

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    dotenv.load_dotenv()
    HF_TOKEN = os.environ.get('HF_TOKEN')

    embedding_dir_path = config['paths']['embedding_path']
    bytes_dir_paths = config['paths']['mp4_paths'].split(',')

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    max_files = None
    if max_files is not None:
        data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', f"topic_model_videos_{max_files}")
    else:
        data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', f"topic_model_videos")
    os.makedirs(data_dir_path, exist_ok=True)

    if not os.path.exists(os.path.join(data_dir_path, 'sample_df.parquet.zstd')):
        embeddings, video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, max_files=max_files)
    else:
        video_df = pl.read_parquet(os.path.join(data_dir_path, 'video_df.parquet.zstd'))
        embeddings = None

    topic_model = ExtendedTopicModel.load(data_dir_path)

    if os.path.exists(os.path.join(data_dir_path, 'umap_model.sav')):
        topic_model.umap_model = joblib.load(os.path.join(data_dir_path, 'umap_model.sav'))

    if os.path.exists(os.path.join(data_dir_path, 'hdbscan_model.sav')):
        topic_model.hdbscan_model = joblib.load(os.path.join(data_dir_path, 'hdbscan_model.sav'))

    reduced_embeddings = np.load(os.path.join(data_dir_path, 'reduced_embeddings.npy'))

    batch_size = 500000
    topics = []
    for idx in tqdm(range(0, len(video_df), batch_size), desc='Transforming videos'):
        batch_video_df = video_df[idx:idx+batch_size]
        # batch_embeddings = embeddings[idx:idx+batch_size]
        batch_umap_embeddings = reduced_embeddings[idx:idx+batch_size]
        if 'desc' in batch_video_df.columns:
            batch_topics, _ = topic_model.transform(batch_video_df['desc'].to_list(), umap_embeddings=batch_umap_embeddings)
        else:
            batch_topics, _ = topic_model.transform(None, umap_embeddings=batch_umap_embeddings, images=batch_video_df['image_path'].to_list())
        topics.extend(batch_topics)
        
    video_df = video_df.with_columns(pl.Series(name='topic', values=topics))
    video_df.write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')

    sample_size = 1000000
    sample_embeddings = reduced_embeddings[np.random.choice(reduced_embeddings.shape[0], sample_size, replace=False)]
    umap_model = umap.UMAP(n_components=2, verbose=True)
    umap_model.fit(sample_embeddings)
    batch_size = 1000000
    embeddings_2d = np.zeros((reduced_embeddings.shape[0], 2))
    for i in range(sample_size, reduced_embeddings.shape[0], batch_size):
        batch_embeddings = reduced_embeddings[i:i+batch_size]
        batch_embeddings = umap_model.transform(batch_embeddings)
        embeddings_2d[i:i+batch_size] = batch_embeddings

    np.save(os.path.join(data_dir_path, '2d_embeddings.npy'), embeddings_2d)

if __name__ == '__main__':
    main()