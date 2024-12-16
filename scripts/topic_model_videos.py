import base64
import configparser
import io
import os
import time

import av
from bertopic import BERTopic
from bertopic.representation import VisualRepresentation, TextGeneration, KeyBERTInspired
from bertopic.representation._mmr import mmr
import dotenv
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
import torch
from tqdm import tqdm
import transformers
from umap import UMAP


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
    embeddings = None
    video_df = None
    num_files = 0
    pbar = tqdm(total=max_files)
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
                # and get that video pat
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

    if embeddings is None and video_df is None:
        raise ValueError("No embeddings found")

    return embeddings, video_df

def get_prompt():
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics.
    <</SYS>>
    """

    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following documents:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the word food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

    [/INST] Environmental impacts of eating meat
    """

    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: '[KEYWORDS]'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
    [/INST]
    """

    prompt = system_prompt + example_prompt + main_prompt
    return prompt

class ExtendedTopicModel(BERTopic):
    def __init__(self, *args, **kwargs):
        if 'nr_repr_docs' in kwargs:
            self.nr_repr_docs = kwargs['nr_repr_docs']
            del kwargs['nr_repr_docs']
        super().__init__(*args, **kwargs)
        

    def _save_representative_docs(self, documents: pd.DataFrame):
        """Save the 3 most representative docs per topic.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Updates:
            self.representative_docs_: Populate each topic with 3 representative docs
        """
        repr_docs, _, _, _ = self._extract_representative_docs(
            self.c_tf_idf_,
            documents,
            self.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.nr_repr_docs,
        )
        self.representative_docs_ = repr_docs

class ExtendedVisualRepresentation(VisualRepresentation):
    def image_to_text(self, documents: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """Convert images to text."""
        # Create image topic embeddings
        topics = documents.Topic.values.tolist()
        images = documents.Image.values.tolist()
        df = pd.DataFrame(np.hstack([np.array(topics).reshape(-1, 1), embeddings]))
        image_topic_embeddings = df.groupby(0).mean().values

        # Extract image centroids
        image_centroids = {}
        unique_topics = sorted(list(set(topics)))
        for topic, topic_embedding in zip(unique_topics, image_topic_embeddings):
            indices = np.array([index for index, t in enumerate(topics) if t == topic])
            top_n = min([self.nr_repr_images, len(indices)])
            indices = mmr(
                topic_embedding.reshape(1, -1),
                embeddings[indices],
                indices,
                top_n=top_n,
                diversity=0.1,
            )
            image_centroids[topic] = indices

        # Extract documents
        documents = pd.DataFrame(columns=["Document", "ID", "Topic", "Image"])
        current_id = 0
        for topic, image_ids in tqdm(image_centroids.items()):
            selected_images = []
            for index in image_ids:
                if isinstance(images[index], str):
                    try:
                        image = Image.open(images[index])
                    except Exception as e:
                        print(f"Error opening image: {e}")
                        continue
                    selected_images.append(image)
                else:
                    selected_images.append(images[index])
        
            text = self._convert_image_to_text(selected_images)

            for doc, image_id in zip(text, image_ids):
                documents.loc[len(documents), :] = [
                    doc,
                    current_id,
                    topic,
                    images[image_id],
                ]
                current_id += 1

            # Properly close images
            if isinstance(images[image_ids[0]], str):
                for image in selected_images:
                    image.close()

        return documents

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
        data_dir_path = os.path.join(this_dir_path, '..', 'data', f"topic_model_videos_{max_files}")
    else:
        data_dir_path = os.path.join(this_dir_path, '..', 'data', f"topic_model_videos")
    os.makedirs(data_dir_path, exist_ok=True)

    embeddings, video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, max_files=max_files)

    method = 'image_caption'
    if method == 'use_desc':
        # Additional ways of representing a topic
        visual_model = VisualRepresentation()
        
        # Make sure to add the `visual_model` to a dictionary
        representation_model = [visual_model]
        # Train our model with images and captions
        topic_model = BERTopic(representation_model=representation_model, verbose=True)
        topics, probs = topic_model.fit_transform(
            documents=video_df['desc'].to_list(), 
            embeddings=embeddings,
            images=video_df['image_path'].to_list()
        )
    elif method == 'image_caption':
        # Additional ways of representing a topic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        im_to_text_model = transformers.pipeline("image-to-text", model='microsoft/git-base', device=device, torch_dtype=torch.float16)
        visual_model = ExtendedVisualRepresentation(image_to_text_model=im_to_text_model)

        num_repr_images = 30

        # Create your representation model
        representation_model = KeyBERTInspired()
        representation_model.image_to_text_model = im_to_text_model
        representation_model.image_to_text = visual_model.image_to_text
        representation_model._chunks = visual_model._chunks
        representation_model.batch_size = visual_model.batch_size
        representation_model.nr_repr_images = num_repr_images

        # Train our model with images
        num_repr_docs = 10
        topic_model = ExtendedTopicModel(
            representation_model=representation_model, 
            embedding_model="paraphrase-MiniLM-L6-v2",
            verbose=True, 
            nr_repr_docs=num_repr_docs
        )
        topics, probs = topic_model.fit_transform(
            documents=None, 
            embeddings=embeddings,
            images=video_df['image_path'].to_list()
        )
    elif method == 'image_caption_llm_sum':
        # Additional ways of representing a topic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        im_to_text_model = transformers.pipeline("image-to-text", model='microsoft/git-base', device=device, torch_dtype=torch.float16)
        visual_model = VisualRepresentation(image_to_text_model=im_to_text_model)

        # Create your representation model
        text_model = transformers.pipeline("text-generation", model='meta-llama/Llama-3.2-1B-Instruct', device='cpu', token=HF_TOKEN, torch_dtype=torch.float16)
        representation_model = TextGeneration(text_model, prompt=get_prompt(), pipeline_kwargs={'max_new_tokens': 20})

        # add image to text model to representation model
        representation_model.image_to_text_model = im_to_text_model
        representation_model.image_to_text = visual_model.image_to_text
        representation_model._chunks = visual_model._chunks
        representation_model.batch_size = visual_model.batch_size

        # Train our model with images
        topic_model = BERTopic(representation_model=representation_model, verbose=True)
        topics, probs = topic_model.fit_transform(
            documents=None, 
            embeddings=embeddings,
            images=video_df['image_path'].to_list()
        )

    video_df = video_df.with_columns(pl.Series(name='topic', values=topics))
    video_df.write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')

    topic_info_df = topic_model.get_topic_info()

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()


    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
    
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    np.save(os.path.join(data_dir_path, 'reduced_embeddings.npy'), reduced_embeddings)

    topic_info_path = os.path.join(data_dir_path, 'topic_info.html')
    topic_info_df.to_html(topic_info_path, formatters={'Visual_Aspect': image_formatter}, escape=False)
    cols = ['Topic', 'Count', 'Name', 'Representation', 'Representative_Docs']
    if 'Visual_Aspect' in topic_info_df.columns:
        topic_info_df['Visual_Aspect_Bytes'] = topic_info_df['Visual_Aspect'].map(lambda i: i.tobytes())
        topic_info_df['Visual_Aspect_Mode'] = topic_info_df['Visual_Aspect'].map(lambda i: i.mode)
        topic_info_df['Visual_Aspect_Size'] = topic_info_df['Visual_Aspect'].map(lambda i: i.size)
        cols += ['Visual_Aspect_Bytes', 'Visual_Aspect_Mode', 'Visual_Aspect_Size']

    topic_info_df[cols].to_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))

if __name__ == '__main__':
    main()