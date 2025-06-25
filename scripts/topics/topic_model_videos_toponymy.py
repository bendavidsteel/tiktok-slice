import base64
import configparser
from datetime import datetime
import io
import os
import pathlib
import time
from typing import List, Any, Optional

import av
import av.error
import datamapplot
import dotenv
import huggingface_hub
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
from tqdm import tqdm
import toponymy
import transformers
from sentence_transformers import SentenceTransformer
from cuml.manifold.umap import UMAP
import vllm
import vllm.v1.engine.exceptions
import vllm.lora.request


def save_first_frame(paths):
    frame_path = paths['image_path']
    video_path = paths['video_path']
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError as e:
        return False
    for frame in container.decode(video=0):
        frame.to_image().save(frame_path)
        return True

def open_with_retries(path, func):
    for i in range(5):
        try:
            return func(path)
        except Exception as e:
            time.sleep(1)
    raise ValueError(f"Error opening {path}")

def get_videos_embeddings(embeddings_dir_path, bytes_dir_paths, hour=None, minute=None, max_files=None):
    """Load video embeddings and metadata - same as original but without max_files limit"""
    video_df = None
    num_files = 0
    
    # Count total files for progress bar
    total_dirs = len([d for d in os.listdir(embeddings_dir_path) 
                     if os.path.isdir(os.path.join(embeddings_dir_path, d))])
    pbar = tqdm(total=total_dirs if max_files is None else max_files)
    
    for dir_name in os.listdir(embeddings_dir_path):
        if hour is not None or  minute is not None:
            dir_time = datetime.fromtimestamp(int(dir_name))
            if hour is not None and dir_time.hour != hour:
                continue
            if minute is not None and dir_time.minute != minute:
                continue
        try:
            batch_embedding_path = os.path.join(embeddings_dir_path, dir_name, 'video_embeddings.npy')
            videos_path = os.path.join(embeddings_dir_path, dir_name, 'videos.parquet.gzip')
            if pathlib.Path(batch_embedding_path).exists() and pathlib.Path(videos_path).exists():
                # with np.load(batch_embedding_path, allow_pickle=True) as f:

                batch_embeddings = np.load(batch_embedding_path)
                if not batch_embeddings.shape:
                    continue

                batch_video_df = pl.read_parquet(videos_path, columns=['id', 'video'])

                if batch_embeddings.shape[0] != len(batch_video_df):
                    continue

                batch_video_df = batch_video_df.with_columns(pl.Series(name='embedding', values=batch_embeddings))

                batch_video_df = batch_video_df.with_columns([
                    pl.col('video').struct.field('desc').alias('desc'),
                    pl.col('video').struct.field('locationCreated').alias('locationCreated'),
                    pl.col('video').struct.field('createTime').alias('createTime'),
                    pl.col('video').struct.field('stats').struct.field('playCount').alias('playCount'),
                ])
                batch_video_df = batch_video_df.drop('video')
                batch_video_df = batch_video_df.with_row_index()
                batch_video_df = batch_video_df.with_columns([
                    pl.col('id').cast(pl.UInt64)
                        .map_elements(lambda i: format(i, '064b'), pl.String)
                        .str.slice(0, 32)
                        .map_elements(lambda s: int(s, 2), pl.UInt64)
                        .alias('timestamp'),
                    pl.lit(bytes_dir_paths).alias('bytes_dir_paths'),
                ])
                batch_video_df = batch_video_df.explode('bytes_dir_paths').rename({'bytes_dir_paths': 'bytes_dir_path'})
                batch_video_df = batch_video_df.with_columns([
                    pl.concat_str([
                        pl.col('bytes_dir_path'),
                        pl.col('timestamp').cast(pl.String),
                        pl.lit('/'),
                        pl.col('id').cast(pl.String),
                        pl.lit('.mp4'),
                    ]).alias('video_path'),
                    pl.concat_str([
                        pl.col('bytes_dir_path'),
                        pl.col('timestamp').cast(pl.String)
                    ]).alias('dir_path')
                ])
                dir_paths = batch_video_df.unique('dir_path')['dir_path'].to_list()
                video_file_paths = [str(p) for p in dir_paths for p in pathlib.Path(p).glob('*.mp4')]
                video_file_df = pl.DataFrame({'video_path': video_file_paths, 'video_path_exists': [True for _ in video_file_paths]})
                batch_video_df = batch_video_df.join(video_file_df, on='video_path', how='left').with_columns(
                    pl.col('video_path_exists').fill_null(False)
                )

                # get videos that have a video path that exists
                batch_video_df = batch_video_df.filter(pl.col('video_path_exists'))
                # batch_video_df = batch_video_df.with_columns(
                #     pl.col('video_path').str.replace('.mp4', '.jpg', literal=True).alias('image_path')
                # )
                # image_file_paths = [str(p) for p in dir_paths for p in pathlib.Path(p).glob('*.jpg')]
                # image_file_df = pl.DataFrame({'image_path': image_file_paths, 'image_path_exists': [True for _ in image_file_paths]})
                # batch_video_df = batch_video_df.join(image_file_df, on='image_path', how='left').with_columns(
                #     pl.col('image_path_exists').fill_null(False)
                # )
                # path_d = batch_video_df.filter(~pl.col('image_path_exists')).select(['image_path', 'video_path']).to_dicts()
                # if path_d:
                #     for path in path_d:
                #         path['image_path_created'] = save_first_frame(path)
                #     created_df = pl.from_dicts(path_d)
                #     batch_video_df = batch_video_df.join(created_df.drop('video_path'), on='image_path', how='left').with_columns(
                #         pl.col('image_path_created').fill_null(False)
                #     ).with_columns(
                #         pl.when(pl.col('image_path_created') & ~pl.col('image_path_exists'))\
                #         .then(pl.lit(True))\
                #         .otherwise(pl.col('image_path_exists')).alias('image_path_exists')
                #     ).drop('image_path_created')
                #     batch_video_df = batch_video_df.filter(pl.col('image_path_exists'))

                if video_df is None:
                    video_df = batch_video_df
                else:
                    video_df = pl.concat([video_df, batch_video_df])

                pbar.update(1)

                if max_files:
                    num_files += 1
                    if num_files == max_files:
                        break

        except Exception as e:
            print(f"Error processing {dir_name}: {e}")

    pbar.close()

    if video_df is None:
        raise ValueError("No embeddings found")

    return video_df

def get_middle_frame(video_path):
    container = av.open(video_path)
    
    # Get video stream and its duration
    video_stream = container.streams.video[0]
    duration = container.duration  # in microseconds
    
    if duration:
        # Calculate middle timestamp (convert to seconds, then to the stream's time_base)
        middle_time_seconds = (duration / av.time_base) / 2
        middle_timestamp = int(middle_time_seconds / video_stream.time_base)
        
        # Seek to the middle of the video
        container.seek(middle_timestamp, stream=video_stream)
    
    # Get the first frame after seeking (which should be near the middle)
    for frame in container.decode(video=0):
        image = frame.to_image()
        break
    
    container.close()
    return image

class VisionVLLM(toponymy.llm_wrappers.AsyncVLLM):
    def caption_image(self, video_paths):
        images = []
        for video_path in video_paths:
            try:
                image = get_middle_frame(video_path)
            except av.error.InvalidDataError as e:
                images.append(None)
                continue
            images.append(image.convert('RGB'))
        question = 'Please give a detailed but concise description of the image.'
        
        prompt = f"<|user|><|image_1|>{question}<|end|><|assistant|>"

        sampling_params = vllm.SamplingParams(
            temperature=0.0,
            max_tokens=64,
        )

        # Since the vision-lora and speech-lora co-exist with the base model,
        # we have to manually specify the path of the lora weights.
        # vision_lora_path = os.path.join(self.llm.model_path, "vision-lora")
        # lora_request = vllm.lora.request.LoRARequest("vision", 1, vision_lora_path)

        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                'image': image
            },
        } for image in images]
        try:
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        except vllm.v1.engine.exceptions.EngineDeadError:
            print("Error generating captions, restarting engine...")
            self.start_engine()
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        texts = [o.outputs[0].text for o in outputs]
        i = 0
        final_texts = []
        for image in images:
            if image is None:
                final_texts.append('')
            else:
                final_texts.append(texts[i])
                i += 1
        return final_texts


def main():
    # Load environment variables
    dotenv.load_dotenv()
    
    # Load configuration - check for both .ini and .yaml files
    config = None
    if os.path.exists('./config/config.ini'):
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        embedding_dir_path = config['paths']['embedding_path']
        bytes_dir_paths = config['paths']['mp4_paths'].split(',')
    else:
        # Fallback to expected paths based on the original script structure
        print("Warning: config.ini not found, using default paths")
        embedding_dir_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings')
        bytes_dir_paths = [os.path.join(os.path.dirname(__file__), '..', 'data', 'videos')]
    
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    # Always use full dataset (no max_files limit)
    data_dir_path = os.path.join(this_dir_path, '..', 'data', 'topic_model_videos_toponymy')
    os.makedirs(data_dir_path, exist_ok=True)

    print("Loading video embeddings and metadata...")
    video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, hour=19, max_files=None)
    
    video_df.write_parquet(os.path.join(data_dir_path, 'video_embeddings.parquet.zstd'), compression='zstd')
    return
    print(f"Loaded {video_df.shape[0]} video embeddings")

    # drop duplicate video embeddings
    video_df = video_df.unique('embedding')
    embeddings = video_df['embedding'].to_numpy()

    print(f"Unique video embeddings: {embeddings.shape[0]}")
    
    # Create low-dimensional representation using UMAP
    print("Creating low-dimensional document representation...")
    umap_model = UMAP(
        metric='cosine',
        random_state=42,
        build_algo="nn_descent", 
        build_kwds={"nnd_do_batch": True, "nnd_n_clusters": 4}
    )
    document_map = umap_model.fit_transform(embeddings, data_on_host=True)
    
    # Initialize embedding model for Toponymy
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create clusterer
    # convert embeddings to float32 for compatibility with Toponymy
    embeddings = embeddings.astype(np.float32)

    print("Creating document clusters...")
    # model_path = huggingface_hub.snapshot_download("microsoft/Phi-4-multimodal-instruct")
    llm_wrapper = VisionVLLM(
        "microsoft/Phi-4-multimodal-instruct", 
        trust_remote_code=True,
        max_num_seqs=2,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        # enable_lora=True,
        # max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={"dynamic_hd": 16},
        limit_mm_per_prompt={'image': 1},
        # disable_mm_preprocessor_cache=False,
        # enforce_eager=Truee 
    )
    clusterer = toponymy.ToponymyClusterer(min_clusters=4, verbose=True)
    clusterer.fit(clusterable_vectors=document_map, embedding_vectors=embeddings, object_to_text_function=llm_wrapper.caption_image, show_progress_bar=True)
    
    # Initialize local LLM wrapper
    
    
    # Create Toponymy topic model
    print("Creating Toponymy topic model...")
    topic_model = toponymy.Toponymy(
        llm_wrapper=llm_wrapper,
        text_embedding_model=embedding_model,
        clusterer=clusterer,
        object_description="TikTok videos",
        corpus_description="collection of TikTok videos with descriptions and visual content"
    )
    
    # Fit the model
    print("Fitting topic model...")
    image_paths = video_df['video_path'].to_list()
    topic_model.fit(image_paths, embeddings, document_map)
    
    
    # Create interactive DataMapPlot visualization
    print("Creating interactive DataMapPlot visualization...")
    
    # Prepare topic names for visualization
    topic_name_vectors = [cluster_layer.topic_name_vector for cluster_layer in topic_model.cluster_layers_]
    
    video_df.select(['id', 'desc', 'locationCreated', 'createTime', 'playCount', 'video_path'])\
        .with_columns([pl.Series(name=f'topic_layer_{i}', values=topic_name_vectors[i]) for i in range(len(topic_name_vectors))])\
        .with_columns(pl.Series(name='map', values=document_map))\
        .write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')


if __name__ == '__main__':
    main()