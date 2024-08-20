import asyncio
import datetime
import json
import os
import re
import subprocess

import asyncssh
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd
import tqdm
import whisper


def transcribe_directory(transcription_model, video_df, dir_path):
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

    video_transcription_path = os.path.join(dir_path, 'video_transcriptions.parquet.gzip')

    add_to_existing = False
    if os.path.exists(video_transcription_path):
        try:
            saved_video_df = pd.read_parquet(video_transcription_path)
            
            saved_video_ids = set(saved_video_df['id'].tolist())
            video_ids = set(video_df['id'].tolist())
            if saved_video_ids == video_ids:
                return
            elif saved_video_ids.issubset(video_ids):
                add_to_existing = True
                video_df = video_df[~video_df['id'].isin(saved_video_ids)]
                host_file_paths = [host_file_path for host_file_path in host_file_paths if os.path.splitext(os.path.basename(host_file_path))[0] in video_df['id'].tolist()]
        except Exception as e:
            print(f"Failed to load previous transcriptions: {e}")

    transcriptions = {}
    for host_file_path in tqdm.tqdm(host_file_paths):
        video_id = re.search(r'(\d+).mp4', os.path.basename(host_file_path)).group(1)
        audio_file_path = os.path.join(dir_path, f"{video_id}.mp3")
        if not os.path.exists(audio_file_path):
            try:
                video = VideoFileClip(host_file_path)
            except OSError:
                continue
            video.audio.write_audiofile(audio_file_path)

        transcription_file_path = os.path.join(dir_path, f"{video_id}.json")
        if not os.path.exists(transcription_file_path):
            transcription = transcription_model.transcribe(audio_file_path)
            transcriptions[video_id] = transcription

    video_df['transcription'] = video_df['id'].map(lambda id: transcriptions.get(id, None))

    if add_to_existing:
        video_df = pd.concat([saved_video_df, video_df], axis=0)

    video_df.to_parquet(video_transcription_path, compression='gzip')

    # delete the files from the host
    # for file_path in host_file_paths:
    #     os.remove(file_path)

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.dirname(this_dir_path)
    host_results_dir_path = os.path.join('/', 'mnt', 'bigone', 'bsteel', 'tiktok', 'data', 'results')
    host_bytes_dir_path = os.path.join('/', 'mnt', 'bigone', 'bsteel', 'tiktok', 'data', 'bytes')

    transcription_model = whisper.load_model("base")

    username = os.environ['USERNAME']
    password = os.environ['PASSWORD']
    host = os.environ['ELITE_HOST']
    server_bytes_dir_path = '/media/bsteel/NAS/TikTok_Hour/mp4s/'
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
        
        transcribe_directory(transcription_model, video_df, host_dir_path)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())

