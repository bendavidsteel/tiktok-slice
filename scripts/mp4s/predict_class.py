import configparser
import datetime
import os
import joblib

import matplotlib as mpl
import numpy as np
import polars as pl
from PIL import Image
import plotly.express as px
import pycountry
import tqdm
import torch
import transformers

from scripts.children.train_classifier import StackingEnsembleClassifier

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        self.model = transformers.AutoModel.from_pretrained("microsoft/xclip-base-patch32", device_map=self.device, torch_dtype=torch.float16)

        self.porn_prompts = [
            "a naked person",
            "a person having sex",
            "people engaging in sexual activity",
            "pornography",
            "oral sex",
            "a person masturbating"
        ]

        self.non_porn_prompts = [
            "a love story",
            "a clothed woman dancing",
            "an animated woman",
            "a video game",
            "an animated movie",
            "a woman in a bikini",
            "a woman in makeup",
            "writing on a screen",
            "a social media post",
            "a person wearing clothes",
            "people clothed",
            "some objects",
            "a landscape",
            "a building",
            "a vehicle",
            "food",
            "a piece of clothing",
            "a piece of furniture"
        ]

        self.violence_prompts = [
            "a person being killed",
            "a person being injured",
            "a person being attacked",
            "violence",
            "someone bleeding",
            "a person being shot",
            "a person being stabbed",
            "a person being raped"
        ]

        self.non_violence_prompts = [
            "a tug of war",
            "a dusty room with dancers",
            "a person crying",
            "breakdancing",
            "martial arts",
            "a blood effect social media face filter",
            "a vaccination",
            "a person in a hospital",
            "a man in a headscarf",
            "a kitchen knife",
            "people hunting",
            "people at night",
            "a person in a room",
            "a selfie",
            "a person smiling",
            "a person laughing",
            "a person eating",
            "a person drinking",
            "a landscape",
            "a building",
            "a vehicle",
            "food",
            "a piece of clothing",
            "a piece of furniture"
        ]

        inputs = tokenizer(
            self.porn_prompts + self.non_porn_prompts + self.violence_prompts + self.non_violence_prompts, 
            padding=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.text_features = self.model.get_text_features(**inputs)

        # load child classifier
        self.child_classifier = joblib.load('./models/stacking_ensemble.joblib')

    def classify(self, embeddings, img_features):
        
        batch_size = 32
        class_probs = pl.DataFrame()
        for i in tqdm.tqdm(range(0, len(embeddings), batch_size), desc="Classifying"):
            class_batch_df = self._classify_batch_video(embeddings[i:i+batch_size], img_features[i:i+batch_size])
            class_probs = pl.concat([class_probs, class_batch_df], how='diagonal_relaxed')
        return class_probs

    def _classify_batch_video(self, embeddings, img_features):
        video_embeds = torch.tensor(embeddings)
        img_features = torch.tensor(img_features)

        video_embeds = video_embeds.to(self.device)
        img_features = img_features.to(self.device)
        
        # Normalize video embeddings
        video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Get logit scale from model
        logit_scale = self.model.logit_scale.exp()
        
        # Create separate groups for each classifier
        porn_prompts_features = self.text_features[:len(self.porn_prompts) + len(self.non_porn_prompts)]
        violence_prompts_features = self.text_features[-len(self.violence_prompts) - len(self.non_violence_prompts):]
        
        batch_size = embeddings.shape[0]
        
        # Process porn classifier
        porn_text_embeds = porn_prompts_features.unsqueeze(0).expand(batch_size, -1, -1)
        porn_text_embeds = porn_text_embeds + self.model.prompts_generator(porn_text_embeds, img_features)
        porn_text_embeds = porn_text_embeds / porn_text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Process violence classifier
        violence_text_embeds = violence_prompts_features.unsqueeze(0).expand(batch_size, -1, -1)
        violence_text_embeds = violence_text_embeds + self.model.prompts_generator(violence_text_embeds, img_features)
        violence_text_embeds = violence_text_embeds / violence_text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate logits for each classifier separately
        porn_logits = torch.einsum("bd,bkd->bk", video_embeds, logit_scale * porn_text_embeds)
        violence_logits = torch.einsum("bd,bkd->bk", video_embeds, logit_scale * violence_text_embeds)
        
        # Process child probabilities
        child_prob = self.child_classifier.predict_proba(embeddings)
        
        # Process porn probabilities
        all_porn_probs = torch.softmax(porn_logits, dim=1)
        pos_porn_prob = torch.mean(all_porn_probs[:, :len(self.porn_prompts)], dim=1)
        neg_porn_prob = torch.mean(all_porn_probs[:, len(self.porn_prompts):], dim=1)
        porn_prob = pos_porn_prob / (pos_porn_prob + neg_porn_prob)
        
        # Process violence probabilities
        all_violence_probs = torch.softmax(violence_logits, dim=1)
        pos_violence_prob = torch.mean(all_violence_probs[:, :len(self.violence_prompts)], dim=1)
        neg_violence_prob = torch.mean(all_violence_probs[:, len(self.violence_prompts):], dim=1)
        violence_prob = pos_violence_prob / (pos_violence_prob + neg_violence_prob)
        
        # Convert to numpy for dataframe creation
        porn_prob = porn_prob.cpu().detach().numpy()
        violence_prob = violence_prob.cpu().detach().numpy()
        
        return pl.DataFrame({
            'child_prob': child_prob,
            'porn_prob': porn_prob,
            'violence_prob': violence_prob
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

    embeddings, img_features, video_df = get_videos_embeddings(embedding_dir_path, max_files=max_files, hour=hour, minute=minute)

    classifier = Classifier()
    class_prob_df = classifier.classify(embeddings, img_features)
    video_df = pl.concat([video_df, class_prob_df], how='horizontal')

    os.makedirs(dir_path, exist_ok=True)
    video_df.write_parquet(os.path.join(dir_path, 'video_class_prob_test.parquet.gzip'))

if __name__ == '__main__':
    main()