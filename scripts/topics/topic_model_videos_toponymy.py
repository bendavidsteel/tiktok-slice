import base64
import configparser
from datetime import datetime
import io
import os
import pathlib
import time
from typing import Any, Callable, Union, Optional, List, Dict, cast

import av
import av.error
import datamapplot
import dotenv
import huggingface_hub
import numpy as np
import pandas as pd
from PIL import Image
import polars as pl
from tqdm.auto import tqdm
import toponymy
import toponymy.embedding_wrappers
import transformers
from sentence_transformers import SentenceTransformer
from cuml.manifold.umap import UMAP
import vllm
import vllm.v1.engine.exceptions
import vllm.lora.request
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                        ChatTemplateContentFormatOption,
                                        apply_hf_chat_template,
                                        apply_mistral_chat_template,
                                        parse_chat_messages,
                                        resolve_chat_template_content_format)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput


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
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError as e:
        return None
    
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
    else:
        # If no frames were decoded, return None
        return None
    
    container.close()
    image = image.convert('RGB')
    return image


class LLM(vllm.LLM):
    def chat(
        self,
        messages: Union[list[ChatCompletionMessageParam],
                        list[list[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = {},
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][] method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                similar to OpenAI schema.
                Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        list_of_messages: list[list[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is list[list[...]]
            list_of_messages = cast(list[list[ChatCompletionMessageParam]],
                                    messages)
        else:
            # messages is list[...]
            list_of_messages = [
                cast(list[ChatCompletionMessageParam], messages)
            ]

        tokenizer = self.get_tokenizer(lora_request)
        model_config = self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        prompts: list[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format=resolved_content_format,
            )

            if isinstance(tokenizer, MistralTokenizer):
                prompt_token_ids = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    **_chat_template_kwargs,
                )
            else:
                prompt_str = apply_hf_chat_template(
                    tokenizer=tokenizer,
                    conversation=conversation,
                    model_config=model_config,
                    **_chat_template_kwargs,
                )
                # Special tokens are already included in chat templates so
                # should not be added by the tokenizer in this case.
                prompt_token_ids = tokenizer.encode(prompt_str,
                                                    add_special_tokens=False,
                                                    **tokenizer_kwargs)

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            if mm_processor_kwargs is not None:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs

            prompts.append(prompt)

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

class VisionVLLM(toponymy.llm_wrappers.AsyncVLLM):
    def __init__(self, model: str, llm_specific_instructions: Optional[str] = None, truncate=True, caption_cache_path: str = None, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._start_engine()
        self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        self.truncate = truncate

        self.caption_cache_path = caption_cache_path
        if os.path.exists(self.caption_cache_path):
            self.caption_cache_df = pl.read_parquet(self.caption_cache_path)
        else:
            self.caption_cache_df = pl.DataFrame({'video_path': [], 'caption': []}, schema={'video_path': pl.String, 'caption': pl.String})
        self.last_saved = len(self.caption_cache_df) if self.caption_cache_df is not None else 0

    async def _call_llm_with_messages(
            self, messages: List[List[dict[str, str]]], temperature: float, max_tokens: int, json_schema: Optional[str] = None
    ) -> List[str]:
        if json_schema:
            guided_decoding_params = vllm.sampling_params.GuidedDecodingParams(json=json_schema)
        else:
            guided_decoding_params = None

        tokenizer_kwargs = {}
        if self.truncate:
            tokenizer_kwargs['max_length'] = self.llm.llm_engine.model_config.max_model_len
            tokenizer_kwargs['truncation'] = True

        sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens, 
            repetition_penalty=1.2,
            guided_decoding=guided_decoding_params
        )
        
        chat_kwargs = {
            'messages': messages,
            'sampling_params': sampling_params,
            'tokenizer_kwargs': tokenizer_kwargs
        }

        try:
            outputs = self.llm.chat(**chat_kwargs)
        except vllm.v1.engine.exceptions.EngineDeadError:
            self._start_engine()  # Restart the engine if it fails
            outputs = self.llm.chat(**chat_kwargs)

        return [output.outputs[0].text for output in outputs]

    def caption_image(self, video_paths):

        # Cache fetch mechanism
        cached_results = {}
        uncached_video_paths = []
        
        for i, video_path in enumerate(video_paths):
            # Check if this video path exists in cache
            cached_row = self.caption_cache_df.filter(pl.col('video_path') == video_path)
            if len(cached_row) > 0:
                # Found in cache, store the result
                cached_results[video_path] = cached_row['caption'][0]
            else:
                # Not in cache, add to list for processing
                uncached_video_paths.append((i, video_path))

        # If all videos are cached, return cached results
        if not uncached_video_paths:
            return [cached_results[path] for path in video_paths]
        
        # Continue with existing logic for uncached videos only
        images = []
        for i, video_path in uncached_video_paths:  # Changed from video_paths
            try:
                image = get_middle_frame(video_path)
                if image is None:
                    continue
                images.append((i, image))
            except Exception as e:
                raise Exception(f"Error processing video {video_path}: {e}")
            
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
        } for i, image in images]
        try:
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        except vllm.v1.engine.exceptions.EngineDeadError:
            print("Error generating captions, restarting engine...")
            self._start_engine()
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        texts = [o.outputs[0].text for o in outputs]
        
        computed_texts = {}
        for (i, image), text in zip(images, texts):
            computed_texts[i] = text

        final_texts = []
        for i, video_path in enumerate(video_paths):
            if video_path in cached_results:
                final_texts.append(cached_results[video_path])
            elif i in computed_texts:
                final_texts.append(computed_texts[i])
            else:
                final_texts.append("")

        # Save captions to cache
        new_captions = pl.DataFrame({
            'video_path': video_paths,
            'caption': final_texts
        })
        self.caption_cache_df = pl.concat([self.caption_cache_df, new_captions])
        self.caption_cache_df = self.caption_cache_df.unique('video_path')
        if len(self.caption_cache_df) - self.last_saved > 100:
            self.caption_cache_df.write_parquet(self.caption_cache_path, compression='zstd')
            self.last_saved = len(self.caption_cache_df)
            print(f"Saved {self.last_saved} captions to cache")

        return final_texts


def main():
    # Load environment variables
    dotenv.load_dotenv()
    
    # Load configuration - check for both .ini and .yaml files
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    embedding_dir_path = config['paths']['embedding_path']
    bytes_dir_paths = config['paths']['mp4_paths'].split(',')
    
    # Always use full dataset (no max_files limit)
    use = '24hour'
    if use == '1hour':
        data_dir_path = os.path.join('.', 'data', 'topic_model_videos_toponymy')
        hour = 19
        minute = None
    elif use == '24hour':
        data_dir_path = os.path.join('.', 'data', 'topic_model_videos_toponymy_24hour')
        hour = None
        minute = 42
    os.makedirs(data_dir_path, exist_ok=True)

    print("Loading video embeddings and metadata...")
    embedding_path = os.path.join(data_dir_path, 'video_embeddings.parquet.zstd')
    if not os.path.exists(embedding_path):
        video_df = get_videos_embeddings(embedding_dir_path, bytes_dir_paths, hour=hour, minute=minute, max_files=None)
        video_df.write_parquet(embedding_path, compression='zstd')
    else:
        video_df = pl.read_parquet(embedding_path)
    
    print(f"Loaded {video_df.shape[0]} video embeddings")

    # drop duplicate video embeddings
    video_df = video_df.unique('embedding')

    umap_path = os.path.join(data_dir_path, 'video_embeddings_umap.parquet.zstd')
    if os.path.exists(umap_path):
        umap_df = pl.read_parquet(umap_path)
        video_df = video_df.join(umap_df, on='id', how='left').filter(pl.col('umap_embedding').is_not_null())
    else:
        print("Calculating UMAP embeddings...")
        umap = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric='cosine',
            verbose=True,
            random_state=42
        )
        embeddings = video_df['embedding'].to_numpy()
        umap_embeddings = umap.fit_transform(embeddings)
        video_df = video_df.with_columns(pl.Series(name='umap_embedding', values=umap_embeddings))
        video_df.write_parquet(umap_path, compression='zstd')
    
    # Initialize embedding model for Toponymy
    print("Initializing embedding model...")
    embedding_model = toponymy.embedding_wrappers.VLLMEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    
    # Create clusterer
    

    print("Creating document clusters...")

    # Initialize local LLM wrapper
    
    # model_path = huggingface_hub.snapshot_download("microsoft/Phi-4-multimodal-instruct")
    llm_wrapper = VisionVLLM(
        "microsoft/Phi-4-multimodal-instruct", 
        caption_cache_path=os.path.join(data_dir_path, 'caption_cache.parquet.zstd'),
        truncate=True,
        trust_remote_code=True,
        max_num_seqs=2,
        max_model_len=4608,
        gpu_memory_utilization=0.9,
        # enable_lora=True,
        # max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={"dynamic_hd": 16},
        limit_mm_per_prompt={'image': 1, 'audio': 0},
        # disable_mm_preprocessor_cache=False,
        # enforce_eager=Truee 
    )

    if len(video_df) > 1000000:
        base_min_cluster_size = 200
        min_clusters = 4
    elif len(video_df) > 100000:
        base_min_cluster_size = 50
        min_clusters = 3
    else:
        base_min_cluster_size = 10
        min_clusters = 2

    clusterer = toponymy.ToponymyClusterer(min_clusters=min_clusters, base_min_cluster_size=base_min_cluster_size, verbose=True)

    # convert embeddings to float32 for compatibility with Toponymy
    embeddings = video_df['embedding'].to_numpy()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)
    umap_embeddings = np.ascontiguousarray(video_df['umap_embedding'].to_numpy(), dtype=np.float64)

    print(f"Unique video embeddings: {embeddings.shape[0]}")

    clusterer.fit(clusterable_vectors=umap_embeddings, embedding_vectors=embeddings, object_to_text_function=llm_wrapper.caption_image, show_progress_bar=True)
    
    
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
    topic_model.fit(image_paths, embeddings, umap_embeddings)
    
    
    # Create interactive DataMapPlot visualization
    print("Creating interactive DataMapPlot visualization...")
    
    # Prepare topic names for visualization
    topic_name_vectors = [cluster_layer.topic_name_vector for cluster_layer in topic_model.cluster_layers_]
    
    video_df.select(['id', 'desc', 'locationCreated', 'createTime', 'playCount', 'video_path'])\
        .with_columns([pl.Series(name=f'topic_layer_{i}', values=topic_name_vectors[i]) for i in range(len(topic_name_vectors))])\
        .with_columns(pl.Series(name='map', values=umap_embeddings))\
        .write_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'), compression='gzip')


if __name__ == '__main__':
    main()