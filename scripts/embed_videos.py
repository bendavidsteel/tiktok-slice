import asyncio
import datetime
import json
import logging
import multiprocessing
import os
import subprocess

import asyncssh
import av
import numpy as np
import pandas as pd
import torch
from transformers import XCLIPVisionModel, XCLIPTextModel, AutoProcessor, AutoTokenizer
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
        raise

class MultiModalBackend:
    def __init__(self):
        self.device = "cuda"
        self.vision_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32", device=self.device, torch_dtype=torch.float16)
        self.vision_model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32", device_map=self.device, torch_dtype=torch.float16)

        # self.text_model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

    def embed_videos(self, video_file_paths, texts=None):
        logger.debug(f"Loading {len(video_file_paths)} videos...")
        with multiprocessing.Pool(min(8, len(video_file_paths))) as p:
            videos = list(p.imap(load_video, video_file_paths))
        
        logger.debug(f"Embedding {len(videos)} videos...")
        # TODO 
        pixel_values = self.vision_processor(videos=videos, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        num_videos, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        outputs = self.vision_model(pixel_values)
        video_last_hidden_state = outputs.last_hidden_state
        batch_size, num_tokens, embed_size = video_last_hidden_state.shape
        video_last_hidden_state = video_last_hidden_state.reshape(num_videos, num_frames, num_tokens, embed_size)
        # TODO another way of gettign the video embeddings?
        video_embeds = torch.mean(video_last_hidden_state, dim=(1,2))
        video_embeds = video_embeds.cpu().detach().numpy()

        if texts is not None:
            raise NotImplementedError()
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
            outputs = self.text_model(**inputs)
            text_last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled (EOS token) states

            return video_embeds, text_last_hidden_state
        else:
            return video_embeds

def embed_directory(embedding_model, video_df, dir_path):
    host_file_paths = [os.path.join(dir_path, server_filename) for server_filename in os.listdir(dir_path) if server_filename.endswith('.mp4')]
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

    embedding_path = os.path.join(dir_path, 'video_embeddings.npy')
    video_path = os.path.join(dir_path, 'videos.parquet.gzip')

    add_to_existing = False
    if os.path.exists(embedding_path) and os.path.exists(video_path):
        try:
            saved_embeddings = np.load(embedding_path)
            saved_video_df = pd.read_parquet(video_path)
            
            if saved_embeddings.shape[0] == saved_video_df.shape[0]:
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
    i = 0
    max_batch_file_size = 5e7
    max_batch_size = 16
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
            batch_embeddings = embedding_model.embed_videos(batch_file_paths)
        except Exception as e:
            print(f"Failed to embed videos: {e}")
            video_df = video_df[~video_df['id'].isin([os.path.splitext(os.path.basename(file_path))[0] for file_path in batch_file_paths])]
            continue
        pbar.update(batch_size)
        if embeddings is None:
            embeddings = batch_embeddings
        else:
            embeddings = np.concatenate([embeddings, batch_embeddings], axis=0)

    if embeddings is None:
        return

    assert embeddings.shape[0] == video_df.shape[0]
    assert embeddings.shape[0] > 0

    if add_to_existing:
        embeddings = np.concatenate([saved_embeddings, embeddings], axis=0)
        video_df = pd.concat([saved_video_df, video_df], axis=0)

    assert embeddings.shape[0] == video_df.shape[0]
    assert embeddings.shape[0] > 0

    with open(embedding_path, 'wb') as f:
        np.save(f, embeddings)

    video_df.to_parquet(video_path, compression='gzip')

    # delete the files from the host
    # for file_path in host_file_paths:
    #     os.remove(file_path)

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.dirname(this_dir_path)
    host_results_dir_path = os.path.join('/', 'mnt', 'bigone', 'bsteel', 'tiktok', 'data', 'results')
    host_bytes_dir_path = os.path.join('/', 'mnt', 'bigone', 'bsteel', 'tiktok', 'data', 'bytes')

    embedding_model = MultiModalBackend()

    # host_dirs = os.listdir(host_bytes_dir_path)

    # for host_dir in host_dirs:
    #     print(f"Embedding videos in {host_dir}")
    #     host_dir_path = os.path.join(host_bytes_dir_path, host_dir)
    #     embed_directory(embedding_model, host_dir_path)

    username = os.environ['USERNAME']
    password = os.environ['PASSWORD']
    host = os.environ['ELITE_HOST']
    server_bytes_dir_path = '/media/bsteel/Elements/tiktok/mp4s/'
    server_videos_dir_path = '~/repos/what-for-where/data/results/'

    connection = await asyncssh.connect(host, username=username, password=password)
    r = await connection.run(f'ls {server_bytes_dir_path}/', check=True)
    server_dirs = r.stdout.split('\n')
    server_dirs = [server_dir for server_dir in server_dirs if server_dir and "." not in server_dir]
    for server_dir in server_dirs:
        print(f"Embedding videos in {server_dir}")
        dir_time = datetime.datetime.fromtimestamp(int(server_dir))
        video_path = os.path.join(dir_time.strftime('%Y_%m_%d'), 'hours', str(dir_time.hour), str(dir_time.minute), str(dir_time.second), 'results.parquet.gzip')
        host_video_path = os.path.join(host_results_dir_path, video_path)
        if not os.path.exists(host_video_path):
            os.makedirs(os.path.dirname(host_video_path), exist_ok=True)
            server_video_path = os.path.join(server_videos_dir_path, video_path)
            try:
                await asyncssh.scp((connection, server_video_path), host_video_path)
            except Exception as e:
                print(f"Failed to copy {server_video_path} to {host_video_path}: {e}")
                continue
        video_df = pd.read_parquet(host_video_path, columns=['result'])
        server_dir_path = os.path.join(server_bytes_dir_path, server_dir)
        host_dir_path = os.path.join(host_bytes_dir_path, server_dir)
        os.makedirs(host_dir_path, exist_ok=True)
        
        r = subprocess.run(f'rsync -avz --include="*.mp4" --exclude="*" {username}@{host}:{server_dir_path}/* {host_dir_path}/', shell=True, capture_output=True)
        
        if r.returncode != 0:
            print(f"Failed to copy files from {server_dir_path} to {host_dir_path}: {r.stderr}")
            continue
        
        embed_directory(embedding_model, video_df, host_dir_path)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
