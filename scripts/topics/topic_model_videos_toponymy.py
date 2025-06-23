import base64
import configparser
import io
import os
import time
from datetime import datetime

import av
import datamapplot
import dotenv
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
import torch
from tqdm import tqdm
import transformers
from umap import UMAP
from vllm import LLM, SamplingParams

# Toponymy imports
from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import vLLMWrapper
from sentence_transformers import SentenceTransformer


def save_first_frame(paths):
    frame_path = paths['image_path']
    video_path = paths['video_path']
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frame.to_image().save(frame_path)
        return

def open_with_retries(path, func):
    for i in range(5):
        try:
            return func(path)
        except Exception as e:
            time.sleep(1)
    raise ValueError(f"Error opening {path}")

def get_videos_embeddings(embeddings_dir_path, bytes_dir_paths, max_files=None):
    """Load video embeddings and metadata - same as original but without max_files limit"""
    embeddings = None
    video_df = None
    num_files = 0
    
    # Count total files for progress bar
    total_dirs = len([d for d in os.listdir(embeddings_dir_path) 
                     if os.path.isdir(os.path.join(embeddings_dir_path, d))])
    pbar = tqdm(total=total_dirs if max_files is None else max_files)
    
    for dir_name in os.listdir(embeddings_dir_path):
        try:
            batch_embedding_path = os.path.join(embeddings_dir_path, dir_name, 'video_embeddings.npy')
            videos_path = os.path.join(embeddings_dir_path, dir_name, 'videos.parquet.gzip')
            if os.path.exists(batch_embedding_path) and os.path.exists(videos_path):
                batch_embeddings = open_with_retries(batch_embedding_path, lambda p: np.load(p, allow_pickle=True))
                if not batch_embeddings.shape:
                    continue

                batch_video_df = open_with_retries(videos_path, lambda p: pl.read_parquet(p, columns=['id', 'video']))

                if batch_embeddings.shape[0] != len(batch_video_df):
                    continue

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
                batch_video_df = batch_video_df.with_columns(
                    pl.concat_str([
                        pl.col('bytes_dir_path'),
                        pl.col('timestamp').cast(pl.String),
                        pl.lit('/'),
                        pl.col('id').cast(pl.String),
                        pl.lit('.mp4'),
                    ]).alias('video_path')
                )
                batch_video_df = batch_video_df.with_columns(pl.col('video_path').map_elements(os.path.exists, return_dtype=pl.Boolean, strategy='threading').alias('video_path_exists'))

                # get videos that have a video path that exists
                batch_video_df = batch_video_df.filter(pl.col('video_path_exists'))
                indexer = batch_video_df['index'].to_numpy()
                if indexer.sum() == 0:
                    continue
                batch_embeddings = batch_embeddings[indexer]
                batch_video_df = batch_video_df.with_columns(
                    pl.col('video_path').str.replace('.mp4', '.jpg', literal=True).alias('image_path')
                )
                batch_video_df = batch_video_df.with_columns(pl.col('image_path').map_elements(os.path.exists, return_dtype=pl.Boolean, strategy='threading').alias('image_path_exists'))
                batch_video_df.filter(~pl.col('image_path_exists')).with_columns(pl.struct([pl.col('video_path'), pl.col('image_path')]).map_elements(save_first_frame, return_dtype=pl.String, strategy='threading'))
                
                assert batch_embeddings.shape[0] == len(batch_video_df)

                if embeddings is None:
                    embeddings = batch_embeddings
                else:
                    embeddings = np.concatenate([embeddings, batch_embeddings])

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

    if embeddings is None and video_df is None:
        raise ValueError("No embeddings found")

    return embeddings, video_df

def get_thumbnail(image_path, size=(150, 150)):
    """Create thumbnail of image for visualization"""
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', size, color='white')

def image_base64(im):
    """Convert image to base64 for HTML display"""
    if isinstance(im, str):
        im = get_thumbnail(im)
    with io.BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    """Format image for HTML table display"""
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def timestamp_to_datetime(timestamp):
    """Convert timestamp to readable datetime string"""
    try:
        # Convert to seconds (assuming timestamp is in microseconds based on the original code)
        dt = datetime.fromtimestamp(timestamp / 1000000)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Unknown"

class LocalLLMWrapper:
    """Custom LLM wrapper for local Phi-4 via vLLM"""
    
    def __init__(self, model_name="microsoft/Phi-4", max_tokens=50, temperature=0.3):
        print(f"Initializing local LLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|end|>", "\n\n"]
        )
    
    def generate(self, prompts):
        """Generate text from prompts"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]
    
    def __call__(self, prompt):
        """Make the wrapper callable"""
        return self.generate(prompt)[0]

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
    embeddings, video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, max_files=None)
    
    print(f"Loaded {embeddings.shape[0]} video embeddings")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    # Create documents list - use video descriptions where available, otherwise use image paths
    documents = []
    images = []
    timestamps = []
    video_ids = []
    locations = []
    valid_indices = []
    
    for idx, row in enumerate(video_df.iter_rows(named=True)):
        desc = row.get('desc')
        image_path = row.get('image_path')
        timestamp = row.get('timestamp')
        video_id = row.get('id')
        location = row.get('locationCreated')
        
        if desc and desc.strip():
            documents.append(desc.strip())
            images.append(image_path)
            timestamps.append(timestamp)
            video_ids.append(video_id)
            locations.append(location if location else "Unknown")
            valid_indices.append(idx)
        elif image_path and os.path.exists(image_path):
            # Use image path as fallback document
            documents.append(f"Video image: {os.path.basename(image_path)}")
            images.append(image_path)
            timestamps.append(timestamp)
            video_ids.append(video_id)
            locations.append(location if location else "Unknown")
            valid_indices.append(idx)
    
    print(f"Using {len(documents)} documents for topic modeling")
    
    # Filter embeddings to match valid documents
    if valid_indices:
        embeddings = embeddings[valid_indices]
        video_df = video_df[valid_indices]
    else:
        raise ValueError("No valid documents found for topic modeling")
    
    # Create low-dimensional representation using UMAP
    print("Creating low-dimensional document representation...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    document_map = umap_model.fit_transform(embeddings)
    
    # Save reduced embeddings for visualization
    np.save(os.path.join(data_dir_path, 'reduced_embeddings.npy'), document_map)
    
    # Initialize embedding model for Toponymy
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create clusterer
    print("Creating document clusters...")
    clusterer = ToponymyClusterer(min_clusters=4)
    clusterer.fit(clusterable_vectors=document_map, embedding_vectors=embeddings)
    
    # Initialize local LLM wrapper
    print("Initializing local LLM (Phi-4)...")
    llm_wrapper = LocalLLMWrapper(model_name="microsoft/Phi-4")
    
    # Create Toponymy topic model
    print("Creating Toponymy topic model...")
    topic_model = Toponymy(
        llm_wrapper=llm_wrapper,
        text_embedding_model=embedding_model,
        clusterer=clusterer,
        object_description="TikTok videos",
        corpus_description="collection of TikTok videos with descriptions and visual content"
    )
    
    # Fit the model
    print("Fitting topic model...")
    topic_model.fit(documents, embeddings, document_map)
    
    # Extract topics and create results
    print("Extracting topic information...")
    
    # Get topic assignments for each document
    topic_assignments = []
    for layer in topic_model.cluster_layers_:
        # For now, use the finest grain clustering (last layer)
        if hasattr(layer, 'labels_'):
            topic_assignments = layer.labels_
            break
    
    if not topic_assignments:
        # Fallback: assign all documents to topic 0
        topic_assignments = [0] * len(documents)
    
    # Add topic assignments to video dataframe
    video_df = video_df.with_columns(pl.Series(name='topic', values=topic_assignments))
    video_df.write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')
    
    # Create topic info dataframe
    topic_info_data = []
    unique_topics = sorted(list(set(topic_assignments)))
    
    print(f"Found {len(unique_topics)} topics")
    
    for topic_id in unique_topics:
        # Get documents in this topic
        topic_docs = [doc for i, doc in enumerate(documents) if topic_assignments[i] == topic_id]
        topic_images = [img for i, img in enumerate(images) if topic_assignments[i] == topic_id]
        
        # Get topic name from Toponymy
        topic_name = "Unknown Topic"
        if hasattr(topic_model, 'topic_names_') and topic_id < len(topic_model.topic_names_):
            topic_name = topic_model.topic_names_[topic_id]
        
        # Sample representative documents and images
        n_repr_docs = min(5, len(topic_docs))
        n_repr_images = min(3, len(topic_images))
        
        repr_docs = topic_docs[:n_repr_docs] if topic_docs else []
        repr_images = topic_images[:n_repr_images] if topic_images else []
        
        # Create representative image for visualization
        visual_aspect = None
        if repr_images:
            try:
                visual_aspect = get_thumbnail(repr_images[0])
            except:
                pass
        
        topic_info_data.append({
            'Topic': topic_id,
            'Count': len(topic_docs),
            'Name': topic_name,
            'Representation': ', '.join(topic_name.split()[:10]),  # First 10 words as keywords
            'Representative_Docs': repr_docs,
            'Visual_Aspect': visual_aspect
        })
    
    # Create topic info DataFrame
    topic_info_df = pd.DataFrame(topic_info_data)
    
    # Create enhanced hover text with timestamps and metadata
    print("Preparing enhanced hover text with timestamps...")
    hover_texts = []
    for i, (doc, ts, vid_id, loc) in enumerate(zip(documents, timestamps, video_ids, locations)):
        datetime_str = timestamp_to_datetime(ts)
        topic_name = topic_info_df[topic_info_df['Topic'] == topic_assignments[i]]['Name'].iloc[0] if topic_assignments[i] < len(unique_topics) else "Unknown"
        
        hover_text = f"""Topic: {topic_name}
Time: {datetime_str}
Location: {loc}
Video ID: {vid_id}
Description: {doc[:100]}{'...' if len(doc) > 100 else ''}"""
        hover_texts.append(hover_text)
    
    # Create interactive DataMapPlot visualization
    print("Creating interactive DataMapPlot visualization...")
    
    # Prepare topic names for visualization
    topic_name_vectors = []
    if hasattr(topic_model, 'cluster_layers_') and topic_model.cluster_layers_:
        for cluster_layer in topic_model.cluster_layers_:
            if hasattr(cluster_layer, 'topic_name_vector'):
                topic_name_vectors.append(cluster_layer.topic_name_vector)
    
    # Create the interactive plot with enhanced hover text
    if topic_name_vectors:
        plot = datamapplot.create_interactive_plot(
            document_map,
            *topic_name_vectors,
            title="TikTok Video Topic Map",
            sub_title="Interactive exploration of video topics using Toponymy - hover for timestamps and details",
            hover_text=hover_texts,
            color_palette="glasbey_dark",
            point_size=2.5
        )
    else:
        # Fallback: create plot with topic assignments
        topic_labels = [f"Topic {t}" for t in topic_assignments]
        plot = datamapplot.create_interactive_plot(
            document_map,
            labels=topic_labels,
            title="TikTok Video Topic Map", 
            sub_title="Interactive exploration of video topics using Toponymy - hover for timestamps and details",
            hover_text=hover_texts,
            color_palette="glasbey_dark",
            point_size=2.5
        )
    
    # Save the interactive plot
    plot_path = os.path.join(data_dir_path, 'interactive_topic_map.html')
    plot.save(plot_path)
    print(f"Interactive plot saved to: {plot_path}")
    
    # Also create a static PNG version
    static_plot_path = os.path.join(data_dir_path, 'topic_map.png')
    datamapplot.create_plot(
        document_map,
        labels=[f"Topic {t}" for t in topic_assignments],
        title="TikTok Video Topic Map",
        figsize=(12, 8)
    )
    import matplotlib.pyplot as plt
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static plot saved to: {static_plot_path}")
    
    # Save topic information
    topic_info_path = os.path.join(data_dir_path, 'topic_info.html')
    topic_info_df.to_html(topic_info_path, formatters={'Visual_Aspect': image_formatter}, escape=False)
    
    # Prepare columns for parquet export
    cols = ['Topic', 'Count', 'Name', 'Representation', 'Representative_Docs']
    parquet_df = topic_info_df[cols].copy()
    
    # Handle visual aspects for parquet export
    if 'Visual_Aspect' in topic_info_df.columns and topic_info_df['Visual_Aspect'].notna().any():
        try:
            parquet_df['Visual_Aspect_Bytes'] = topic_info_df['Visual_Aspect'].apply(
                lambda x: x.tobytes() if x is not None else b''
            )
            parquet_df['Visual_Aspect_Mode'] = topic_info_df['Visual_Aspect'].apply(
                lambda x: x.mode if x is not None else ''
            )
            parquet_df['Visual_Aspect_Size'] = topic_info_df['Visual_Aspect'].apply(
                lambda x: x.size if x is not None else (0, 0)
            )
        except Exception as e:
            print(f"Warning: Could not save visual aspects to parquet: {e}")
    
    parquet_df.to_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))
    
    # Save metadata with timestamps for further analysis
    metadata_df = pd.DataFrame({
        'video_id': video_ids,
        'timestamp': timestamps,
        'datetime': [timestamp_to_datetime(ts) for ts in timestamps],
        'location': locations,
        'topic': topic_assignments,
        'topic_name': [topic_info_df[topic_info_df['Topic'] == t]['Name'].iloc[0] if t < len(unique_topics) else "Unknown" for t in topic_assignments],
        'document': documents
    })
    metadata_df.to_parquet(os.path.join(data_dir_path, 'video_metadata_with_topics.parquet.gzip'))
    
    print(f"\nTopic modeling complete! Results saved to: {data_dir_path}")
    print(f"Total documents processed: {len(documents)}")
    print(f"Topics discovered: {len(unique_topics)}")
    print(f"Interactive visualization: {plot_path}")
    print(f"Metadata with timestamps: {os.path.join(data_dir_path, 'video_metadata_with_topics.parquet.gzip')}")
    print("\nTopic Summary:")
    for idx, row in enumerate(topic_info_data):
        print(f"Topic {row['Topic']}: {row['Name']} ({row['Count']} documents)")

if __name__ == '__main__':
    main()