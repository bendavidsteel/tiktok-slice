import configparser
import datetime
import json
import logging
import multiprocessing
import os

import av
import numpy as np
import pandas as pd
import polars as pl
import torch
from transformers import XCLIPVisionModel, XCLIPTextModel, AutoProcessor, AutoModel
import tqdm

from embed_videos import MultiModalBackend

logger = logging.getLogger(__name__)

def embed_directory(embedding_model, video_df, write_dir_path, read_dir_paths):
    host_file_paths = []
    for read_dir_path in read_dir_paths:
        host_file_paths += [os.path.join(read_dir_path, server_filename) for server_filename in os.listdir(read_dir_path) if server_filename.endswith('.mp4')]
    host_file_paths = sorted(host_file_paths)
    byte_video_ids = [os.path.splitext(os.path.basename(host_file_path))[0] for host_file_path in host_file_paths]

    # get video data for each video
    video_df['return'] = video_df['result'].map(lambda r: r['return'])
    video_df['id'] = video_df['return'].map(lambda r: r['id'] if r and 'id' in r else None)
    video_df = video_df[['return', 'id']].rename(columns={'return': 'video'})
    video_df = video_df[video_df['id'].map(lambda id: id is not None)]
    meta_video_ids = video_df['id'].tolist()
    video_ids = list(set(byte_video_ids).intersection(set(meta_video_ids)))

    host_file_paths = [host_file_path for host_file_path in host_file_paths if os.path.splitext(os.path.basename(host_file_path))[0] in video_ids]
    bytes_video_id_order = [os.path.splitext(os.path.basename(host_file_path))[0] for host_file_path in host_file_paths]
    video_df = video_df[video_df['id'].isin(video_ids)]
    # reorder based on host_file_paths
    video_df = video_df.set_index('id')
    video_df = video_df.loc[bytes_video_id_order]
    video_df = video_df.reset_index()

    if len(host_file_paths) == 0:
        return

    embedding_path = os.path.join(write_dir_path, 'video_embeddings.npy')
    img_features_path = os.path.join(write_dir_path, 'img_features.npy')
    video_path = os.path.join(write_dir_path, 'videos.parquet.gzip')

    add_to_existing = False
    if os.path.exists(embedding_path) and os.path.exists(img_features_path) and os.path.exists(video_path):
        try:
            saved_embeddings = np.load(embedding_path)
            saved_img_features = np.load(img_features_path)
            saved_video_df = pd.read_parquet(video_path)
            
            if saved_embeddings.shape[0] == saved_video_df.shape[0] and saved_img_features.shape[0] == saved_video_df.shape[0]:
                saved_video_ids = set(saved_video_df['id'].tolist())
                video_ids = set(video_df['id'].tolist())
        except Exception as e:
            print(f"Failed to load embeddings: {e}")

    embeddings = None
    img_features = None
    i = 0
    max_batch_file_size = 2e8
    max_batch_size = 32
    pbar = tqdm.tqdm(total=len(host_file_paths))
    while i < len(host_file_paths):
        batch_file_size = 0
        batch_size = 0
        batch_file_paths = []
        while i < len(host_file_paths) and batch_file_size < max_batch_file_size and batch_size < max_batch_size:
            file_stats = os.stat(host_file_paths[i])
            batch_file_paths.append(host_file_paths[i])
            batch_file_size += file_stats.st_size
            batch_size += 1
            i += 1

        # embed the videos
        try:
            batch_embeddings, batch_img_features, processed_video_file_paths = embedding_model.embed_videos(batch_file_paths)
        except Exception as e:
            print(f"Failed to embed video batch: {e}")
            video_df = video_df[~video_df['id'].isin([os.path.splitext(os.path.basename(file_path))[0] for file_path in batch_file_paths])]
            continue

        failed_file_paths = [file_path for file_path in batch_file_paths if file_path not in processed_video_file_paths]
        if len(failed_file_paths) > 0:
            print(f"Failed to embed some videos: {failed_file_paths}")
            assert len(video_df[video_df['id'].isin([os.path.splitext(os.path.basename(file_path))[0] for file_path in failed_file_paths])]) == len(failed_file_paths)
            video_df = video_df[~video_df['id'].isin([os.path.splitext(os.path.basename(file_path))[0] for file_path in failed_file_paths])]
            
        pbar.update(batch_size)
        if embeddings is None:
            embeddings = batch_embeddings
        else:
            embeddings = np.concatenate([embeddings, batch_embeddings], axis=0)

        if img_features is None:
            img_features = batch_img_features
        else:
            img_features = np.concatenate([img_features, batch_img_features], axis=0)

    saved_video_df = pl.from_pandas(saved_video_df)
    video_df = pl.from_pandas(video_df)

    saved_video_df = saved_video_df.with_row_index()
    video_df = video_df.with_row_index()

    assert np.allclose(saved_embeddings, embeddings[saved_video_df.join(video_df, on='id')['index_right'].to_numpy()], 1e-2)


def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.dirname(this_dir_path)

    embedding_model = MultiModalBackend()

    bytes_dir_paths = config['paths']['mp4_paths'].split(',')
    videos_dir_path = config['paths']['result_path']
    embedding_dir_path = config['paths']['embedding_path']

    server_dirs = [dir_name for byte_dir_path in bytes_dir_paths for dir_name in os.listdir(byte_dir_path)]
    server_dirs = [server_dir for server_dir in server_dirs if server_dir and "." not in server_dir]
    server_dirs = set(server_dirs)
    for server_dir in server_dirs:
        print(f"Embedding videos in {server_dir}")
        try:
            dir_time = datetime.datetime.fromtimestamp(int(server_dir))
            video_path = os.path.join(dir_time.strftime('%Y_%m_%d'), 'hours', str(dir_time.hour), str(dir_time.minute), str(dir_time.second), 'results.parquet.gzip')
            video_path = os.path.join(videos_dir_path, video_path)
            video_df = pd.read_parquet(video_path, columns=['result'])
            read_dir_paths = []
            for byte_dir_path in bytes_dir_paths:
                read_dir_path = os.path.join(byte_dir_path, server_dir)
                if os.path.exists(read_dir_path):
                    read_dir_paths.append(read_dir_path)

            write_dir_path = os.path.join(embedding_dir_path, server_dir)
        
            embed_directory(embedding_model, video_df, write_dir_path, read_dir_paths)
        except Exception as e:
            print(f"Failed to embed videos: {e}, in {server_dir}")

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
