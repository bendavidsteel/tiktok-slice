import base64
import configparser
import io
import os
import time
from typing import List, Tuple, Union

import av
from bertopic import BERTopic
from bertopic.backend._utils import select_backend
from bertopic.cluster import BaseCluster
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
from bertopic.representation import VisualRepresentation, TextGeneration, KeyBERTInspired
from bertopic.representation._mmr import mmr
from bertopic._utils import check_embeddings_shape, MyLogger, check_is_fitted
import dotenv
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import transformers
import umap

from topic_model_videos import ExtendedTopicModel, ExtendedVisualRepresentation, get_videos_embeddings

logger = MyLogger()
logger.configure("WARNING")

def save_precursors(video_df: pl.DataFrame, embeddings, data_dir_path, HF_TOKEN):
    method = 'image_caption'

    pre_df = video_df.select(['desc', 'image_path'])

    pre_df.write_parquet(os.path.join(data_dir_path, 'sample_df.parquet.zstd'), compression='zstd')
    if method == 'use_desc':
        # Additional ways of representing a topic
        visual_model = VisualRepresentation()
        
        # Make sure to add the `visual_model` to a dictionary
        representation_model = [visual_model]
        # Train our model with images and captions
        topic_model = ExtendedTopicModel(representation_model=representation_model, verbose=True)

    elif method == 'image_caption':
        num_repr_images = 40
        num_repr_docs = 40
        
        topic_model = ExtendedTopicModel(
            representation_model=None,#representation_model, 
            embedding_model="paraphrase-MiniLM-L6-v2",
            verbose=True, 
            nr_repr_docs=num_repr_docs
        )

    elif method == 'image_caption_llm_sum':
        # Train our model with images
        topic_model = ExtendedTopicModel(representation_model=None, verbose=True)
        
    umap_embeddings = topic_model.umap_model.fit_transform(embeddings)
    np.save(os.path.join(data_dir_path, 'sample_umap_embeddings.npy'), umap_embeddings)

    # Save the model
    joblib.dump(topic_model.umap_model, os.path.join(data_dir_path, 'umap_model.sav'))
    return topic_model.umap_model


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

    embeddings, video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, max_files=max_files)

    video_df.select(['desc', 'image_path']).write_parquet(os.path.join(data_dir_path, 'video_df.parquet.zstd'), compression='zstd')

    sample_size = 1000000
    if len(video_df) > sample_size:
        if 'index' in video_df.columns:
            video_df = video_df.drop('index')
        video_df = video_df.with_row_index()
        sample_video_df = video_df.sample(sample_size)
        sample_embeddings = embeddings[sample_video_df['index'].to_numpy()]
    else:
        sample_video_df = video_df
        sample_embeddings = embeddings
    np.save(os.path.join(data_dir_path, 'sample_embeddings.npy'), sample_embeddings)
    umap_model = save_precursors(sample_video_df, sample_embeddings, data_dir_path, HF_TOKEN)

    batch_size = 500000
    reduced_embeddings = None
    for idx in tqdm(range(0, len(video_df), batch_size), desc='Transforming videos'):
        batch_embeddings = embeddings[idx:idx+batch_size]
        umap_embeddings = umap_model.transform(batch_embeddings)
        if reduced_embeddings is None:
            reduced_embeddings = umap_embeddings
        else:
            reduced_embeddings = np.concatenate([reduced_embeddings, umap_embeddings])

    np.save(os.path.join(data_dir_path, 'reduced_embeddings.npy'), reduced_embeddings)

if __name__ == '__main__':
    main()