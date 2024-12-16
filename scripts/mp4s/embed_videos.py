import configparser
import datetime
import json
import logging
import multiprocessing
import os

import av
import numpy as np
import pandas as pd
import torch
from transformers import XCLIPVisionModel, XCLIPTextModel, AutoProcessor, AutoModel
import tqdm

logger = logging.getLogger(__name__)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return [x.to_ndarray(format="rgb24") for x in frames]

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    # converted_len = int(clip_len * frame_sample_rate)
    # if seg_len <= converted_len:
    #     raise ValueError(f"Video is too short to sample {clip_len} frames with sample rate {frame_sample_rate}")
    # end_idx = np.random.randint(converted_len, seg_len)
    # start_idx = end_idx - converted_len
    start_idx = 0
    end_idx = seg_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_video(video_file_path):
    try:
        with av.open(video_file_path) as container:
            # sample 16 frames
            indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)
        return video
    except Exception as e:
        os.remove(video_file_path)
        return None
        raise Exception(e, f"Failed to load video: {video_file_path}")
        

class MultiModalBackend:
    def __init__(self):
        self.device = "cuda"
        self.vision_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32", device=self.device, torch_dtype=torch.float16)
        self.vision_model = AutoModel.from_pretrained("microsoft/xclip-base-patch32", device_map=self.device, torch_dtype=torch.float16)

        # self.text_model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

    def embed_videos(self, video_file_paths, texts=None):
        logger.debug(f"Loading {len(video_file_paths)} videos...")
        with multiprocessing.Pool(min(8, len(video_file_paths))) as p:
            videos = list(p.imap(load_video, video_file_paths))

        processed_video_file_paths = [video_file_path for video_file_path, video in zip(video_file_paths, videos) if video is not None]
        videos = [video for video in videos if video is not None]
        
        logger.debug(f"Embedding {len(videos)} videos...")
        # padding videos to the same length
        vid_num_frames = [len(video) for video in videos]
        if not all([num_frames == vid_num_frames[0] for num_frames in vid_num_frames]):
            max_frames = max(vid_num_frames)
            videos = [video + [video[-1]] * (max_frames - len(video)) for video in videos]
            
        inputs = self.vision_processor(videos=videos, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        pixel_values = inputs["pixel_values"]
        output_attentions = self.vision_model.config.output_attentions
        output_hidden_states = self.vision_model.config.output_hidden_states
        return_dict = self.vision_model.config.use_return_dict

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=False,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.vision_model.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.vision_model.mit(
            cls_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        video_embeds = mit_outputs[1]

        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.vision_model.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.vision_model.prompts_visual_projection
        img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
        img_features = img_features.mean(dim=1, keepdim=False)

        video_embeds = video_embeds.cpu().detach().numpy()
        img_features = img_features.cpu().detach().numpy()

        return video_embeds, img_features, processed_video_file_paths

def embed_directory(embedding_model, video_df, write_dir_path, read_dir_paths):
    host_file_paths = []
    for read_dir_path in read_dir_paths:
        host_file_paths += [os.path.join(read_dir_path, server_filename) for server_filename in os.listdir(read_dir_path) if server_filename.endswith('.mp4')]
    host_file_paths = sorted(host_file_paths)
    host_file_paths = host_file_paths[200:]
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
                if saved_video_ids == video_ids:
                    return
                elif saved_video_ids.issubset(video_ids):
                    add_to_existing = True
                    video_df = video_df[~video_df['id'].isin(saved_video_ids)]
                    host_file_paths = [host_file_path for host_file_path in host_file_paths if os.path.splitext(os.path.basename(host_file_path))[0] in video_df['id'].tolist()]
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

    if embeddings is None:
        return

    if embeddings.shape[0] != len(video_df):
        raise ValueError(f"Embeddings shape {embeddings.shape} does not match video_df length {len(video_df)}")
    assert embeddings.shape[0] > 0

    if add_to_existing:
        embeddings = np.concatenate([saved_embeddings, embeddings], axis=0)
        img_features = np.concatenate([saved_img_features, img_features], axis=0)
        video_df = pd.concat([saved_video_df, video_df], axis=0)


    assert embeddings.shape[0] == video_df.shape[0]
    assert embeddings.shape[0] > 0

    os.makedirs(write_dir_path, exist_ok=True)

    with open(embedding_path, 'wb') as f:
        np.save(f, embeddings)

    with open(img_features_path, 'wb') as f:
        np.save(f, img_features)

    video_df.to_parquet(video_path, compression='gzip')

    # delete the files from the host
    # for file_path in host_file_paths:
    #     os.remove(file_path)

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
        
        try:
            embed_directory(embedding_model, video_df, write_dir_path, read_dir_paths)
        except Exception as e:
            print(f"Failed to embed videos: {e}, in {server_dir}")

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
