import configparser
import datetime
import os
import joblib

import numpy as np
import polars as pl
import tqdm

from train_classifier import LogisticRegressionClassifier

def get_videos_embeddings(embeddings_dir_path, max_files=None, hour=None, minute=None):
    embeddings = None
    img_features = None
    video_df = None
    num_files = 0
    day = 10
    pbar = tqdm.tqdm(total=max_files)
    for dir_name in os.listdir(embeddings_dir_path):
        dir_time = datetime.datetime.fromtimestamp(int(dir_name))
        if dir_time.day != day:
            continue
        if hour is not None and dir_time.hour != hour:
            continue
        if minute is not None and dir_time.minute != minute:
            continue
        try:
            filenames = os.listdir(os.path.join(embeddings_dir_path, dir_name))
            if 'video_embeddings.npy' in filenames and 'videos.parquet.gzip' in filenames and 'img_features.npy' in filenames:
                batch_embeddings = np.load(os.path.join(embeddings_dir_path, dir_name, 'video_embeddings.npy'), allow_pickle=True)
                batch_img_features = np.load(os.path.join(embeddings_dir_path, dir_name, 'img_features.npy'), allow_pickle=True)
                if not batch_embeddings.shape:
                    continue

                batch_video_df = pl.read_parquet(os.path.join(embeddings_dir_path, dir_name, 'videos.parquet.gzip'), columns=['id', 'video'])

                if batch_embeddings.shape[0] != len(batch_video_df):
                    continue

                batch_video_df = batch_video_df.with_columns([
                    pl.col('video').struct.field('desc').alias('desc'),
                    pl.col('video').struct.field('locationCreated').alias('locationCreated'),
                    pl.col('video').struct.field('createTime').alias('createTime'),
                ])
                batch_video_df = batch_video_df.drop('video')
                
                assert batch_embeddings.shape[0] == len(batch_video_df)

                if embeddings is None:
                    embeddings = batch_embeddings
                else:
                    embeddings = np.concatenate([embeddings, batch_embeddings])

                if img_features is None:
                    img_features = batch_img_features
                else:
                    img_features = np.concatenate([img_features, batch_img_features])

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
            print(f"Error with {dir_name}: {e}")
            continue

    if embeddings is None and video_df is None:
        raise ValueError("No embeddings found")

    return embeddings, img_features, video_df


class Classifier:
    def __init__(self):
        # load child classifier
        self.classifier = joblib.load('./models/logistic_regression.joblib')

    def classify(self, embeddings):
        
        batch_size = 32
        class_probs = pl.DataFrame()
        for i in tqdm.tqdm(range(0, len(embeddings), batch_size), desc="Classifying"):
            class_batch_df = self._classify_batch_video(embeddings[i:i+batch_size])
            class_probs = pl.concat([class_probs, class_batch_df], how='diagonal_relaxed')
        return class_probs

    def _classify_batch_video(self, embeddings):
        # Process child probabilities
        prob = self.classifier.predict_proba(embeddings)
        
        return pl.DataFrame({
            'aif_aigc_prob': prob,
        })

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    embedding_dir_path = config['paths']['embedding_path']
    bytes_dir_paths = config['paths']['mp4_paths'].split(',')

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    max_files = None
    use = '24hour'
    if use == 'all':
        hour = None
        minute = None
    elif use == '24hour':
        hour = None
        minute = 42
    elif use == '1hour':
        minute = None
        hour = 17
    dir_path = os.path.join('.', 'data', 'stats', use)

    classifier = Classifier()

    video_df = pl.read_parquet('./data/topic_model_videos_toponymy_24hour/video_embeddings.parquet.zstd', columns=['id', 'locationCreated', 'createTime', 'embedding'])
    embeddings = video_df['embedding'].to_numpy()
    # embeddings, img_features, video_df = get_videos_embeddings(embedding_dir_path, max_files=max_files, hour=hour, minute=minute)

    
    class_prob_df = classifier.classify(embeddings)
    video_df = pl.concat([video_df, class_prob_df], how='horizontal')

    threshold = 0.3
    video_df = video_df.with_columns((pl.col('aif_aigc_prob') > threshold).alias('aif_aigc_pred'))

    os.makedirs(dir_path, exist_ok=True)
    video_df.select(['aif_aigc_prob', 'aif_aigc_pred', 'id', 'locationCreated', 'createTime']).write_parquet(os.path.join(dir_path, 'video_aif_aigc_prob.parquet.zstd'))

if __name__ == '__main__':
    main()