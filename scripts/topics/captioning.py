import os

import numpy as np
import polars as pl
import torch
import transformers

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

class CLIPNounCaptioning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        self.model = transformers.AutoModel.from_pretrained("microsoft/xclip-base-patch32", device_map=self.device, torch_dtype=torch.float16)

        # More comprehensive prompts
        self.nouns = [
            'man',
            'woman',
            'child',
            'family',
            'car',
            'food',
            'dog',
            'cat',
            'bird',
            'tree',
            'flower',
            'house'
        ]

        inputs = tokenizer(self.nouns, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.text_features = self.model.get_text_features(**inputs)

    def classify_child_presence(self, embeddings, img_features):
        
        batch_size = 32
        classifications = []
        for i in range(0, len(embeddings), batch_size):
            classifications.extend(self._classify_batch_video(embeddings[i:i+batch_size], img_features[i:i+batch_size]))

        classifications = [[{'generated_text': classification}] for classification in classifications]
        return classifications

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
        noun_idxs = torch.argmax(probs, axis=1)
    
        return [self.nouns[i] for i in noun_idxs]

    def __call__(self, images):
        image_df = pl.DataFrame({'image_path': images})
        image_df = image_df.with_columns(pl.col('image_path').str.split('/').alias('path_parts'))
        image_df = image_df.with_columns([
            pl.col('path_parts').list.get(-1).alias('file_name'),
            pl.col('path_parts').list.get(-2).alias('dir_name'),
        ])
        image_df = image_df.with_columns(pl.col('file_name').str.replace('.jpg', '', literal=True).alias('id'))
        selected_embeddings = None
        selected_img_features = None
        for dir_name in image_df['dir_name'].unique().to_numpy():
            video_path = os.path.join('.', 'data', 'embeddings', dir_name, 'videos.parquet.gzip')
            video_df = pl.read_parquet(video_path).with_row_index()
            embeddings = np.load(os.path.join('.', 'data', 'embeddings', dir_name, 'video_embeddings.npy'))
            img_features = np.load(os.path.join('.', 'data', 'embeddings', dir_name, 'img_features.npy'))
            dir_image_df = image_df.filter(pl.col('dir_name') == dir_name)
            idx = video_df.join(dir_image_df, on='id', how='right').unique('id')['index'].to_numpy()
            dir_selected_embeddings = embeddings[idx]
            dir_selected_img_features = img_features[idx]
            if selected_embeddings is None:
                selected_embeddings = dir_selected_embeddings
                selected_img_features = dir_selected_img_features
            else:
                selected_embeddings = np.concatenate([selected_embeddings, dir_selected_embeddings])
                selected_img_features = np.concatenate([selected_img_features, dir_selected_img_features])
        assert selected_embeddings.shape[0] == len(images)
        return self.classify_child_presence(selected_embeddings, selected_img_features)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class VLMCaptioning:
    def __init__(self):
        path = "OpenGVLab/InternVL2_5-1B"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.model = transformers.AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()

    def __call__(self, images):
        question = '<image>\nPlease give a detailed but concise description of the image.'
        generation_config = dict(max_new_tokens=256, do_sample=True)
        responses = []
        for image in images:
            try:
                pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                responses.append([{ 'generated_text': response }])
            except Exception as e:
                responses.append([{ 'generated_text': '' }])
        return responses