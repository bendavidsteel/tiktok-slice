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
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_video(video_file_path):
    with av.open(video_file_path) as container:
        # sample 16 frames
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)
    return video

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
    host_file_paths = host_file_paths[200:]
    byte_video_ids = [os.path.splitext(os.path.basename(host_file_path))[0] for host_file_path in host_file_paths]

    # get video data for each video
    video_df['return'] = video_df['result'].map(lambda r: r['return'])
    video_df['id'] = video_df['return'].map(lambda r: r['id'])
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

    embeddings = None
    i = 0
    max_batch_file_size = 5e7
    max_batch_size = 3
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
        batch_embeddings = embedding_model.embed_videos(batch_file_paths)
        pbar.update(batch_size)
        if embeddings is None:
            embeddings = batch_embeddings
        else:
            embeddings = np.concatenate([embeddings, batch_embeddings], axis=0)
        print(f"Embedding size: {embeddings.nbytes / 1e6} MB")

    with open(os.path.join(dir_path, f'video_embeddings.npy'), 'wb') as f:
        np.save(f, embeddings)

    video_df.to_parquet(os.path.join(dir_path, 'videos.parquet.gzip'), compression='gzip')

    # delete the files from the host
    for file_path in host_file_paths:
        os.remove(file_path)

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.dirname(this_dir_path)
    host_bytes_dir_path = os.path.join(root_dir_path, 'data', 'bytes')
    host_results_dir_path = os.path.join(root_dir_path, 'data', 'results')

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
    server_dirs = [server_dir for server_dir in server_dirs if "." not in server_dir]
    for server_dir in server_dirs:
        print(f"Embedding videos in {server_dir}")
        dir_time = datetime.datetime.fromtimestamp(int(server_dir))
        video_path = os.path.join(dir_time.strftime('%Y_%m_%d'), 'hours', str(dir_time.hour), str(dir_time.minute), str(dir_time.second), '0', 'results.parquet.gzip')
        host_video_path = os.path.join(host_results_dir_path, video_path)
        if not os.path.exists(host_video_path):
            os.makedirs(os.path.dirname(host_video_path), exist_ok=True)
            server_video_path = os.path.join(server_videos_dir_path, video_path)
            try:
                await asyncssh.scp((connection, server_video_path), host_video_path)
            except:
                print(f"Failed to copy {server_video_path} to {host_video_path}")
                continue
        video_df = pd.read_parquet(host_video_path, columns=['result'])
        server_dir_path = os.path.join(server_bytes_dir_path, server_dir)
        host_dir_path = os.path.join(host_bytes_dir_path, server_dir)
        os.makedirs(host_dir_path, exist_ok=True)
        
        r = subprocess.run(f'rsync -avz --include="*.mp4" --exclude="*" {username}@{host}:{server_dir_path}/* {host_dir_path}/', shell=True, capture_output=True)
        
        if r.returncode != 0:
            raise Exception(f"Failed to copy files from {server_dir_path} to {host_dir_path}")
        
        embed_directory(embedding_model, video_df, host_dir_path)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
