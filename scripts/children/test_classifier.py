import configparser
import datetime
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
import plotly.express as px
import pycountry
from sklearn.metrics import f1_score
import tqdm
import torch
import transformers


def get_videos_embeddings(embeddings_dir_path):
    embeddings = None
    img_features = None
    video_df = None
    filenames = os.listdir(embeddings_dir_path)
    if 'video_embeddings.npy' in filenames and 'videos.parquet.gzip' in filenames and 'img_features.npy' in filenames:
        batch_embeddings = np.load(os.path.join(embeddings_dir_path, 'video_embeddings.npy'), allow_pickle=True)
        batch_img_features = np.load(os.path.join(embeddings_dir_path, 'img_features.npy'), allow_pickle=True)
        if not batch_embeddings.shape:
            raise ValueError(f"Embeddings shape is empty")

        batch_video_df = pl.read_parquet(os.path.join(embeddings_dir_path, 'videos.parquet.gzip'))

        if batch_embeddings.shape[0] != len(batch_video_df):
            raise ValueError(f"Embeddings shape {batch_embeddings.shape[0]} does not match number of videos {len(batch_video_df)}")
        
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
            video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')
    else:
        raise ValueError(f"Missing files in {embeddings_dir_path}")



    return embeddings, img_features, video_df


class Classifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        self.model = transformers.AutoModel.from_pretrained("microsoft/xclip-base-patch32", device_map=self.device, torch_dtype=torch.float16)

        # More comprehensive prompts
        self.child_prompts = [
            "a boy",
            "a girl",
            "a child",
            "a baby",
            "a child with family",
            "children with family"
        ]
        
        non_child_prompts = [
            "a social media post",
            "some text saying family",
            "a man alone",
            "a woman alone",
            "men",
            "women",
            "an adult",
            "an image of objects",
            "a landscape",
            "a building",
            "a vehicle",
            "food",
            "a piece of clothing",
            "a piece of furniture"
        ]

        inputs = tokenizer(self.child_prompts + non_child_prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.text_features = self.model.get_text_features(**inputs)

    def classify_child_presence(self, embeddings, img_features):
        
        batch_size = 32
        child_prob = []
        for i in tqdm.tqdm(range(0, len(embeddings), batch_size), desc="Classifying"):
            child_prob.extend(self._classify_batch_video(embeddings[i:i+batch_size], img_features[i:i+batch_size]).tolist())
        return child_prob

    def _classify_batch_video(self, embeddings, img_features):
        video_embeds = torch.tensor(embeddings)
        img_features = torch.tensor(img_features)

        video_embeds = video_embeds.to(self.device)
        img_features = img_features.to(self.device)

        text_embeds = self.text_features
        batch_size = embeddings.shape[0]
        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        text_embeds = text_embeds + self.model.prompts_generator(text_embeds, img_features)

        # normalized features
        video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_video = torch.einsum("bd,bkd->bk", video_embeds, logit_scale * text_embeds)
        logits_per_text = logits_per_video.T

        # Average probabilities across prompts for each category
        probs = torch.softmax(logits_per_video, dim=1)
        child_prob = torch.mean(probs[:, :len(self.child_prompts)], axis=1)
        non_child_prob = torch.mean(probs[:, len(self.child_prompts):], axis=1)
    
        child_prob = child_prob / (child_prob + non_child_prob)
        return child_prob.cpu().detach().numpy()

def get_dir_probs(classifier, dir_path):
    embeddings, img_features, video_df = get_videos_embeddings(dir_path)
    child_prob = classifier.classify_child_presence(embeddings, img_features)
    return child_prob

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    classifier = Classifier()

    embedding_dir_path = os.path.join('.', 'data', 'children_val_set', 'child_videos')
    child_probs = get_dir_probs(classifier, embedding_dir_path)

    embedding_dir_path = os.path.join('.', 'data', 'children_val_set', 'non_child_videos')
    non_child_probs = get_dir_probs(classifier, embedding_dir_path)

    
    

    combined_probs = np.concatenate((child_probs, non_child_probs))
    true_labels = np.concatenate((np.ones(len(child_probs)), np.zeros(len(non_child_probs))))

    # get f1 score
    thresholds = np.linspace(0.3, 0.9, 20)
    f1s = []
    for threshold in thresholds:
        predictions = combined_probs > threshold
        f1 = f1_score(true_labels, predictions)
        f1s.append(f1)
    fig, ax = plt.subplots()
    ax.plot(thresholds, f1s)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    fig.savefig('./figs/child_thres_f1.png')

if __name__ == '__main__':
    main()