import json
import os

from moviepy.video.io.VideoFileClip import VideoFileClip
import tqdm
import whisper

def main():
    model = whisper.load_model("base")

    video_mp3 = []

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')
    bytes_dir_path = os.path.join(data_dir_path, 'results', 'bytes')
    audio_dir_path = os.path.join(data_dir_path, 'results', 'audio')
    transcription_dir_path = os.path.join(data_dir_path, 'results', 'transcriptions')

    if not os.path.exists(audio_dir_path):
        os.makedirs(audio_dir_path)
    if not os.path.exists(transcription_dir_path):
        os.makedirs(transcription_dir_path)

    bytes_filenames = os.listdir(bytes_dir_path)
    for byte_filename in tqdm.tqdm(bytes_filenames):
        video_id = byte_filename.split(".")[0]
        audio_file_path = os.path.join(audio_dir_path, f"{video_id}.mp3")
        if not os.path.exists(audio_file_path):
            try:
                video = VideoFileClip(os.path.join(bytes_dir_path, byte_filename))
            except OSError:
                continue
            video.audio.write_audiofile(audio_file_path)

        transcription_file_path = os.path.join(transcription_dir_path, f"{video_id}.json")
        if not os.path.exists(transcription_file_path):
            transcription = model.transcribe(audio_file_path)
            with open(transcription_file_path, 'w') as f:
                json.dump(transcription, f)

if __name__ == '__main__':
    main()