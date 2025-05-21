import base64
import configparser
import io
import joblib
import os
from pathlib import Path
import time
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable

import av
from bertopic import BERTopic
from bertopic._bertopic import TopicMapper
from bertopic.backend import BaseEmbedder
from bertopic.backend._utils import select_backend
from bertopic.cluster import BaseCluster
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import VisualRepresentation, TextGeneration, KeyBERTInspired, BaseRepresentation
from bertopic.representation._mmr import mmr
from bertopic.representation._visual import get_concat_tile_resize
from bertopic._utils import check_embeddings_shape, MyLogger, check_is_fitted, check_documents_type
import bertopic._save_utils as save_utils
import dotenv
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import transformers
import umap

from captioning import CLIPNounCaptioning, VLMCaptioning

logger = MyLogger()
logger.configure("WARNING")

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
                batch_embeddings = np.load(batch_embedding_path)
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


def _create_model_from_files(
    cls,
    topics: Mapping[str, Any],
    params: Mapping[str, Any],
    tensors: Mapping[str, np.array],
    ctfidf_tensors: Mapping[str, Any] = None,
    ctfidf_config: Mapping[str, Any] = None,
    images: Mapping[int, Any] = None,
    warn_no_backend: bool = True,
):
    """Create a BERTopic model from a variety of inputs.

    Arguments:
        topics: A dictionary containing topic metadata, including:
                - Topic representations, labels, sizes, custom labels, etc.
        params: BERTopic-specific hyperparams, including HF embedding_model ID
                if given.
        tensors: The topic embeddings
        ctfidf_tensors: The c-TF-IDF representations
        ctfidf_config: The config for CountVectorizer and c-TF-IDF
        images: The images per topic
        warn_no_backend: Whether to warn the user if no backend is given
    """
    from sentence_transformers import SentenceTransformer

    params["n_gram_range"] = tuple(params["n_gram_range"])

    if ctfidf_config is not None:
        ngram_range = ctfidf_config["vectorizer_model"]["params"]["ngram_range"]
        ctfidf_config["vectorizer_model"]["params"]["ngram_range"] = tuple(ngram_range)

    params["n_gram_range"] = tuple(params["n_gram_range"])
    ctfidf_config

    # Select HF model through SentenceTransformers
    try:
        embedding_model = select_backend(SentenceTransformer(params["embedding_model"]))
    except:  # noqa: E722
        embedding_model = BaseEmbedder()

        if warn_no_backend:
            logger.warning(
                "You are loading a BERTopic model without explicitly defining an embedding model."
                " If you want to also load in an embedding model, make sure to use"
                " `BERTopic.load(my_model, embedding_model=my_embedding_model)`."
            )

    if params.get("embedding_model") is not None:
        del params["embedding_model"]

    # Prepare our empty sub-models
    empty_dimensionality_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()

    # Fit BERTopic without actually performing any clustering
    topic_model = cls(
        embedding_model=embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=empty_cluster_model,
        **params,
    )
    topic_model.topic_embeddings_ = tensors["topic_embeddings"].numpy()
    topic_model.topic_representations_ = {int(key): val for key, val in topics["topic_representations"].items()}
    topic_model.topics_ = topics["topics"]
    topic_model.topic_sizes_ = {int(key): val for key, val in topics["topic_sizes"].items()}
    topic_model.custom_labels_ = topics["custom_labels"]

    if topics.get("topic_aspects"):
        topic_aspects = {}
        for aspect, values in topics["topic_aspects"].items():
            if aspect != "Visual_Aspect":
                topic_aspects[aspect] = {int(topic): value for topic, value in values.items()}
        topic_model.topic_aspects_ = topic_aspects

        if images is not None:
            topic_model.topic_aspects_["Visual_Aspect"] = images

    # Topic Mapper
    topic_model.topic_mapper_ = TopicMapper([0])
    topic_model.topic_mapper_.mappings_ = topics["topic_mapper"]

    if ctfidf_tensors is not None:
        topic_model.c_tf_idf_ = csr_matrix(
            (
                ctfidf_tensors["data"],
                ctfidf_tensors["indices"],
                ctfidf_tensors["indptr"],
            ),
            shape=ctfidf_tensors["shape"],
        )

        # CountVectorizer
        topic_model.vectorizer_model = CountVectorizer(**ctfidf_config["vectorizer_model"]["params"])
        topic_model.vectorizer_model.vocabulary_ = ctfidf_config["vectorizer_model"]["vocab"]

        # ClassTfidfTransformer
        topic_model.ctfidf_model.reduce_frequent_words = ctfidf_config["ctfidf_model"]["reduce_frequent_words"]
        topic_model.ctfidf_model.bm25_weighting = ctfidf_config["ctfidf_model"]["bm25_weighting"]
        idf = ctfidf_tensors["diag"].numpy()
        topic_model.ctfidf_model._idf_diag = sp.diags(
            idf, offsets=0, shape=(len(idf), len(idf)), format="csr", dtype=np.float64
        )
    return topic_model

class ExtendedTopicModel(BERTopic):
    def __init__(
        self,
        language: str = "english",
        top_n_words: int = 10,
        n_gram_range: Tuple[int, int] = (1, 1),
        min_topic_size: int = 10,
        nr_topics: Union[int, str] = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        seed_topic_list: List[List[str]] = None,
        zeroshot_topic_list: List[str] = None,
        zeroshot_min_similarity: float = 0.7,
        embedding_model=None,
        umap_model = None,
        hdbscan_model = None,
        vectorizer_model = None,
        ctfidf_model = None,
        representation_model: BaseRepresentation = None,
        verbose: bool = False,
        nr_repr_docs: int = 3
    ):
        self.nr_repr_docs = nr_repr_docs
        super().__init__(
            language=language, 
            top_n_words=top_n_words,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            low_memory=low_memory,
            calculate_probabilities=calculate_probabilities,
            seed_topic_list=seed_topic_list,
            zeroshot_topic_list=zeroshot_topic_list,
            zeroshot_min_similarity=zeroshot_min_similarity,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            verbose=verbose,
        )

    def fit_transform(
        self,
        documents: List[str],
        embeddings: np.ndarray = None,
        umap_embeddings: np.ndarray = None,
        images: List[str] = None,
        y: Union[List[int], np.ndarray] = None,
    ) -> Tuple[List[int], Union[np.ndarray, None]]:
        """Fit the models on a collection of documents, generate topics,
        and return the probabilities and topic per document.

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            images: A list of paths to the images to fit on or the images themselves
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.

        Examples:
        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings)
        ```
        """
        if documents is not None:
            check_documents_type(documents)
            if umap_embeddings is not None:
                check_embeddings_shape(embeddings, documents)

        doc_ids = range(len(documents)) if documents is not None else range(len(images))
        documents = pd.DataFrame({"Document": documents, "ID": doc_ids, "Topic": None, "Image": images})

        # Extract embeddings
        if embeddings is None and umap_embeddings is None:
            logger.info("Embedding - Transforming documents to embeddings.")
            self.embedding_model = select_backend(self.embedding_model, language=self.language, verbose=self.verbose)
            embeddings = self._extract_embeddings(
                documents.Document.values.tolist(),
                images=images,
                method="document",
                verbose=self.verbose,
            )
            logger.info("Embedding - Completed \u2713")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(
                    self.embedding_model, language=self.language, verbose=self.verbose
                )

        # Guided Topic Modeling
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)

        # Reduce dimensionality and fit UMAP model
        if umap_embeddings is None:
            umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Zero-shot Topic Modeling
        if self._is_zeroshot():
            documents, embeddings, assigned_documents, assigned_embeddings = self._zeroshot_topic_modeling(
                documents, embeddings
            )

            # Filter UMAP embeddings to only non-assigned embeddings to be used for clustering
            if len(documents) > 0:
                umap_embeddings = self.umap_model.transform(embeddings)

        if len(documents) > 0:
            # Cluster reduced embeddings
            documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)
            if self._is_zeroshot() and len(assigned_documents) > 0:
                documents, embeddings = self._combine_zeroshot_topics(
                    documents, embeddings, assigned_documents, assigned_embeddings
                )
        else:
            # All documents matches zero-shot topics
            documents = assigned_documents
            embeddings = assigned_embeddings

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Create documents from images if we have images only
        if documents.Document.values[0] is None:
            custom_documents = self._images_to_text(documents, embeddings)

            # Extract topics by calculating c-TF-IDF
            self._extract_topics(custom_documents, embeddings=embeddings)
            self._create_topic_vectors(documents=documents, embeddings=embeddings)

            # Reduce topics
            if self.nr_topics:
                custom_documents = self._reduce_topics(custom_documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(custom_documents)
        else:
            # Extract topics by calculating c-TF-IDF
            self._extract_topics(documents, embeddings=embeddings, verbose=self.verbose)

            # Reduce topics
            if self.nr_topics:
                documents = self._reduce_topics(documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(documents)

        # In the case of zero-shot topics, probability will come from cosine similarity,
        # and the HDBSCAN model will be removed
        if self._is_zeroshot() and len(assigned_documents) > 0:
            self.hdbscan_model = BaseCluster()
            sim_matrix = cosine_similarity(embeddings, np.array(self.topic_embeddings_))

            if self.calculate_probabilities:
                self.probabilities_ = sim_matrix
            else:
                self.probabilities_ = np.max(sim_matrix, axis=1)
        else:
            self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, self.probabilities_

    def transform(
        self,
        documents: Union[str, List[str]],
        embeddings: np.ndarray = None,
        umap_embeddings: np.ndarray = None,
        images: List[str] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """After having fit a model, use transform to predict new instances.

        Arguments:
            documents: A single document or a list of documents to predict on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model.
            images: A list of paths to the images to predict on or the images themselves

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution which is returned by default.
                           If `calculate_probabilities` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.

        Examples:
        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic().fit(docs)
        topics, probs = topic_model.transform(docs)
        ```

        If you want to use your own embeddings:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic().fit(docs, embeddings)
        topics, probs = topic_model.transform(docs, embeddings)
        ```
        """
        check_is_fitted(self)
        check_embeddings_shape(embeddings, documents)

        if isinstance(documents, str) or documents is None:
            documents = [documents]

        if embeddings is None and umap_embeddings is None:
            embeddings = self._extract_embeddings(documents, images=images, method="document", verbose=self.verbose)

        # Check if an embedding model was found
        if embeddings is None and umap_embeddings is None:
            raise ValueError(
                "No embedding model was found to embed the documents."
                "Make sure when loading in the model using BERTopic.load()"
                "to also specify the embedding model."
            )

        # Transform without hdbscan_model and umap_model using only cosine similarity
        elif type(self.hdbscan_model) == BaseCluster:
            logger.info("Predicting topic assignments through cosine similarity of topic and document embeddings.")
            sim_matrix = cosine_similarity(embeddings, np.array(self.topic_embeddings_))
            predictions = np.argmax(sim_matrix, axis=1) - self._outliers

            if self.calculate_probabilities:
                probabilities = sim_matrix
            else:
                probabilities = np.max(sim_matrix, axis=1)

        # Transform with full pipeline
        else:
            logger.info("Dimensionality - Reducing dimensionality of input embeddings.")
            if umap_embeddings is None:
                umap_embeddings = self.umap_model.transform(embeddings)
            logger.info("Dimensionality - Completed \u2713")

            # Extract predictions and probabilities if it is a HDBSCAN-like model
            logger.info("Clustering - Approximating new points with `hdbscan_model`")
            if is_supported_hdbscan(self.hdbscan_model):
                predictions, probabilities = hdbscan_delegator(
                    self.hdbscan_model, "approximate_predict", umap_embeddings
                )

                # Calculate probabilities
                if self.calculate_probabilities:
                    logger.info("Probabilities - Start calculation of probabilities with HDBSCAN")
                    probabilities = hdbscan_delegator(self.hdbscan_model, "membership_vector", umap_embeddings)
                    logger.info("Probabilities - Completed \u2713")
            else:
                predictions = self.hdbscan_model.predict(umap_embeddings)
                probabilities = None
            logger.info("Cluster - Completed \u2713")

            # Map probabilities and predictions
            probabilities = self._map_probabilities(probabilities, original_topics=True)
            predictions = self._map_predictions(predictions)
        return predictions, probabilities


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

    @classmethod
    def load(cls, path: str, embedding_model=None):
        """Loads the model from the specified path or directory.

        Arguments:
            path: Either load a BERTopic model from a file (`.pickle`) or a folder containing
                  `.safetensors` or `.bin` files.
            embedding_model: Additionally load in an embedding model if it was not saved
                             in the BERTopic model file or directory.

        Examples:
        ```python
        BERTopic.load("model_dir")
        ```

        or if you did not save the embedding model:

        ```python
        BERTopic.load("model_dir", embedding_model="all-MiniLM-L6-v2")
        ```
        """
        file_or_dir = Path(path)

        # Load from Pickle
        if file_or_dir.is_file():
            with open(file_or_dir, "rb") as file:
                if embedding_model:
                    topic_model = joblib.load(file)
                    topic_model.embedding_model = select_backend(embedding_model, verbose=topic_model.verbose)
                else:
                    topic_model = joblib.load(file)
                return topic_model

        # Load from directory or HF
        if file_or_dir.is_dir():
            topics, params, tensors, ctfidf_tensors, ctfidf_config, images = save_utils.load_local_files(file_or_dir)
        elif "/" in str(path):
            topics, params, tensors, ctfidf_tensors, ctfidf_config, images = save_utils.load_files_from_hf(path)
        else:
            raise ValueError("Make sure to either pass a valid directory or HF model.")
        topic_model = _create_model_from_files(
            cls,
            topics,
            params,
            tensors,
            ctfidf_tensors,
            ctfidf_config,
            images,
            warn_no_backend=(embedding_model is None),
        )

        # Replace embedding model if one is specifically chosen
        if embedding_model is not None:
            topic_model.embedding_model = select_backend(embedding_model, verbose=topic_model.verbose)

        return topic_model

class ExtendedVisualRepresentation(VisualRepresentation):
    def __init__(
        self,
        nr_repr_images: int = 9,
        nr_samples: int = 500,
        image_height: Tuple[int, int] = 600,
        image_squares: bool = False,
        image_to_text_model = None,
        batch_size: int = 32,
    ):
        self.nr_repr_images = nr_repr_images
        self.nr_samples = nr_samples
        self.image_height = image_height
        self.image_squares = image_squares

        # Text-to-image model
        self.image_to_text_model = image_to_text_model
        self.batch_size = batch_size

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
                if isinstance(self.image_to_text_model, transformers.Pipeline) or isinstance(self.image_to_text_model, VLMCaptioning):
                    if isinstance(images[index], str):
                        image_path = images[index]
                        try:
                            if not os.path.exists(image_path):
                                video_path = image_path.replace('.jpg', '.mp4')
                                save_first_frame({'image_path': image_path, 'video_path': video_path})
                            with Image.open(images[index]) as image:
                                image.load()
                                selected_images.append(image.copy())
                        except Exception as e:
                            print(f"Error opening image: {e}")
                            continue
                    else:
                        selected_images.append(images[index])
                elif isinstance(self.image_to_text_model, CLIPNounCaptioning):
                    selected_images.append(images[index])
                else:
                    raise ValueError("Image to text model not recognized.")
        
            text = self._convert_image_to_text(selected_images)

            for doc, image_id in zip(text, image_ids):
                documents.loc[len(documents), :] = [
                    doc,
                    current_id,
                    topic,
                    images[image_id],
                ]
                current_id += 1

        return documents

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            representative_images: Representative images per topic
        """
        # Extract image ids of most representative documents
        images = documents["Image"].values.tolist()
        (_, _, _, repr_docs_ids) = topic_model._extract_representative_docs(
            c_tf_idf,
            documents,
            topics,
            nr_samples=self.nr_samples,
            nr_repr_docs=self.nr_repr_images,
        )
        unique_topics = sorted(list(topics.keys()))

        # Combine representative images into a single representation
        representative_images = {}
        for topic in tqdm(unique_topics):
            # Get and order represetnative images
            sliced_examplars = repr_docs_ids[topic + topic_model._outliers]
            sliced_examplars = [sliced_examplars[i : i + 3] for i in range(0, len(sliced_examplars), 3)]

            images_to_combine = []
            for sub_indices in sliced_examplars:
                subs = []
                for index in sub_indices:
                    if isinstance(images[index], str):
                        image_path = images[index]
                        try:
                            if not os.path.exists(image_path):
                                video_path = image_path.replace('.jpg', '.mp4')
                                save_first_frame({'image_path': image_path, 'video_path': video_path})
                            with Image.open(images[index]) as image:
                                image.load()
                                subs.append(image.copy())
                        except Exception as e:
                            print(f"Error opening image: {e}")
                            continue
                    else:
                        subs.append(images[index])
                images_to_combine.append(subs)
            images_to_combine = [sub for sub in images_to_combine if len(sub) > 0]

            # Concatenate representative images
            representative_image = get_concat_tile_resize(images_to_combine, self.image_height, self.image_squares)
            representative_images[topic] = representative_image

            # Make sure to properly close images
            if isinstance(images[0], str):
                for image_list in images_to_combine:
                    for image in image_list:
                        image.close()

        return representative_images

def train_topic_model(video_df, embeddings, data_dir_path, HF_TOKEN, umap_embeddings=None):
    method = 'image_caption'
    if method == 'use_desc':
        # Additional ways of representing a topic
        visual_model = ExtendedVisualRepresentation()
        
        # Make sure to add the `visual_model` to a dictionary
        representation_model = {
            'Visual_Aspect': visual_model
        }
        # Train our model with images and captions
        topic_model = ExtendedTopicModel(representation_model=representation_model, verbose=True)
        docs = video_df['desc'].to_list() if 'desc' in video_df.columns else ['a'] * len(video_df)
        topics, probs = topic_model.fit_transform(
            documents=docs, 
            embeddings=embeddings,
            umap_embeddings=umap_embeddings,
            images=video_df['image_path'].to_list()
        )
    elif method == 'image_caption':
        # Additional ways of representing a topic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # im_to_text_model = transformers.pipeline("image-to-text", model='Salesforce/blip2-flan-t5-xl-coco', device=device, torch_dtype=torch.float16)
        # im_to_text_model = CLIPNounCaptioning()
        im_to_text_model = VLMCaptioning()
        
        num_repr_images = 40
        num_repr_docs = 40
        
        visual_model = ExtendedVisualRepresentation(image_to_text_model=im_to_text_model, batch_size=4, nr_repr_images=num_repr_images)

        # Create your representation model
        representation_model = KeyBERTInspired(nr_repr_docs=num_repr_docs)
        representation_model.image_to_text_model = im_to_text_model
        representation_model.image_to_text = visual_model.image_to_text
        representation_model._chunks = visual_model._chunks
        representation_model.batch_size = visual_model.batch_size
        representation_model.nr_repr_images = num_repr_images

        # Train our model with images
        
        topic_model = ExtendedTopicModel(
            representation_model=representation_model, 
            embedding_model="paraphrase-MiniLM-L6-v2",
            verbose=True, 
            nr_repr_docs=num_repr_docs,
            min_topic_size=max(int(0.0002 * len(video_df)), 2),
            # nr_topics=200
        )
        topics, probs = topic_model.fit_transform(
            documents=None, 
            embeddings=embeddings,
            umap_embeddings=umap_embeddings,
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
        topic_model = ExtendedTopicModel(representation_model=representation_model, verbose=True)
        topics, probs = topic_model.fit_transform(
            documents=None, 
            embeddings=embeddings,
            umap_embeddings=umap_embeddings,
            images=video_df['image_path'].to_list()
        )

    topic_info_df = topic_model.get_topic_info()

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()


    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    topic_info_path = os.path.join(data_dir_path, 'topic_info.html')
    topic_info_df.to_html(topic_info_path, formatters={'Visual_Aspect': image_formatter}, escape=False)
    cols = ['Topic', 'Count', 'Name', 'Representation', 'Representative_Docs']
    if 'Visual_Aspect' in topic_info_df.columns:
        topic_info_df['Visual_Aspect_Bytes'] = topic_info_df['Visual_Aspect'].map(lambda i: i.tobytes())
        topic_info_df['Visual_Aspect_Mode'] = topic_info_df['Visual_Aspect'].map(lambda i: i.mode)
        topic_info_df['Visual_Aspect_Size'] = topic_info_df['Visual_Aspect'].map(lambda i: i.size)
        cols += ['Visual_Aspect_Bytes', 'Visual_Aspect_Mode', 'Visual_Aspect_Size']

    topic_info_df[cols].to_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))

    try:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        topic_model.save(data_dir_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    except Exception as e:
        print(f"Error saving model: {e}")

    # save hdbscan model
    joblib.dump(topic_model.hdbscan_model, os.path.join(data_dir_path, 'hdbscan_model.sav'))

    return topic_model

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

    sample_size = 100000
    if not os.path.exists(os.path.join(data_dir_path, 'sample_df.parquet.zstd')):
        embeddings, video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, max_files=max_files)

        
        if len(video_df) > sample_size:
            if 'index' not in video_df.columns:
                video_df = video_df.with_row_index()
            sample_video_df = video_df.sample(sample_size)
            sample_embeddings = embeddings[sample_video_df['index'].to_numpy()]
        else:
            sample_video_df = video_df
            sample_embeddings = embeddings
        sample_umap_embeddings = None
    else:
        sample_video_df = pl.read_parquet(os.path.join(data_dir_path, 'sample_df.parquet.zstd'))
        sample_embeddings = np.load(os.path.join(data_dir_path, 'sample_embeddings.npy'))
        sample_umap_embeddings = np.load(os.path.join(data_dir_path, 'sample_umap_embeddings.npy'))
        video_df = pl.read_parquet(os.path.join(data_dir_path, 'video_df.parquet.zstd'))
        embeddings = None

    if len(sample_video_df) != sample_size:
        if 'index' in sample_video_df.columns:
            sample_video_df = sample_video_df.drop('index')
        sample_video_df = sample_video_df.with_row_index()
        sample_video_df = sample_video_df.sample(sample_size)
        sample_embeddings = sample_embeddings[sample_video_df['index'].to_numpy()]
        sample_umap_embeddings = sample_umap_embeddings[sample_video_df['index'].to_numpy()]

    topic_model = train_topic_model(sample_video_df, sample_embeddings, data_dir_path, HF_TOKEN, umap_embeddings=sample_umap_embeddings)

    if os.path.exists(os.path.join(data_dir_path, 'umap_model.sav')):
        topic_model.umap_model = joblib.load(os.path.join(data_dir_path, 'umap_model.sav'))

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
            batch_topics, _ = topic_model.transform(None, embeddings=batch_embeddings, umap_embeddings=batch_umap_embeddings, images=batch_video_df['image_path'].to_list())
        topics.extend(batch_topics)
        
    video_df = video_df.with_columns(pl.Series(name='topic', values=topics))
    video_df.write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')

    sample_size = 200000
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