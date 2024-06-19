import base64
import io
import os

import av
from bertopic import BERTopic
from bertopic.representation import VisualRepresentation
import dotenv
import numpy as np
import pandas as pd

def get_video_path(video):
    data_dir_path = os.environ['DATA_DIR_PATH']
    bytes_dir_path = os.path.join(data_dir_path, 'bytes')
    video_id = video['id']
    timestamp = int(format(int(video_id), '064b')[:32], 2)
    return os.path.join(bytes_dir_path, str(timestamp), f"{video_id}.mp4")

def save_first_frame(video_path):
    frame_path = video_path.replace('.mp4', '.jpg')
    if os.path.exists(frame_path):
        return frame_path
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frame.to_image().save(video_path.replace('.mp4', '.jpg'))
        break
    return video_path.replace('.mp4', '.jpg')

def main():
    dotenv.load_dotenv()
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    bytes_dir_path = os.path.join(os.environ['DATA_DIR_PATH'], 'bytes')
    embeddings = None
    video_df = None
    for dir_name in os.listdir(bytes_dir_path):
        filenames = os.listdir(os.path.join(bytes_dir_path, dir_name))
        if 'video_embeddings.npy' in filenames and 'videos.parquet.gzip' in filenames:
            batch_embeddings = np.load(os.path.join(bytes_dir_path, dir_name, 'video_embeddings.npy'), allow_pickle=True)
            if not batch_embeddings.shape:
                continue

            batch_video_df = pd.read_parquet(os.path.join(bytes_dir_path, dir_name, 'videos.parquet.gzip'))

            if batch_embeddings.shape[0] != batch_video_df.shape[0]:
                continue

            batch_video_df['video_path'] = batch_video_df['video'].map(get_video_path)
            indexer = batch_video_df['video_path'].apply(lambda p: os.path.exists(p))
            if indexer.sum() == 0:
                continue
            batch_embeddings = batch_embeddings[indexer]
            batch_video_df = batch_video_df[indexer]
            batch_video_df['image'] = batch_video_df['video_path'].map(save_first_frame)

            assert batch_embeddings.shape[0] == len(batch_video_df)

            if embeddings is None:
                embeddings = batch_embeddings
            else:
                embeddings = np.concatenate([embeddings, batch_embeddings])

            if video_df is None:
                video_df = batch_video_df
            else:
                video_df = pd.concat([video_df, batch_video_df])

    # Additional ways of representing a topic
    visual_model = VisualRepresentation()

    # Make sure to add the `visual_model` to a dictionary
    representation_model = {
        "Visual_Aspect":  visual_model,
    }

    # Train our model with images only
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    topics, probs = topic_model.fit_transform(
        documents=video_df['video'].map(lambda v: v['desc']).tolist(), 
        embeddings=embeddings,
        images=video_df['image'].tolist()
    )

    video_df['topic'] = topics
    video_df.to_parquet(os.path.join(this_dir_path, '..', 'data', 'video_topics.parquet.gzip'))

    topic_info_df = topic_model.get_topic_info()

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()


    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    topic_info_path = os.path.join(this_dir_path, '..', 'data', 'topic_info.html')
    topic_info_df.to_html(topic_info_path, formatters={'Visual_Aspect': image_formatter}, escape=False)
    topic_info_df[['Topic', 'Count', 'Name', 'Representation', 'Representative_Docs']].to_parquet(os.path.join(this_dir_path, '..', 'data', 'topic_info.parquet.gzip'))

if __name__ == '__main__':
    main()