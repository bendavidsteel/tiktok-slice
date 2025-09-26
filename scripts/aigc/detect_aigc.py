import datetime
import os
import pathlib

import av
import peft
from PIL import Image
import polars as pl
from safetensors.torch import load_file as safe_load_file
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def get_middle_frame(video_path):
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError as e:
        return None
    
    # Get video stream and its duration
    video_stream = container.streams.video[0]
    duration = container.duration  # in microseconds
    
    if duration:
        # Calculate middle timestamp (convert to seconds, then to the stream's time_base)
        middle_time_seconds = (duration / av.time_base) / 2
        middle_timestamp = int(middle_time_seconds / video_stream.time_base)
        
        # Seek to the middle of the video
        container.seek(middle_timestamp, stream=video_stream)
    
    # Get the first frame after seeking (which should be near the middle)
    for frame in container.decode(video=0):
        image = frame.to_image()
        break
    else:
        # If no frames were decoded, return None
        return None
    
    container.close()
    image = image.convert('RGB')
    return image


class ImageDataset(Dataset):
    """Dataset that loads every image (recursively) in a folder."""
    def __init__(self, video_df, processor):
        self.processor = processor
        self.video_df = video_df

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        video_path = self.video_df['video_path'][idx]
        try:
            image = get_middle_frame(video_path)
        except Exception:
            image = Image.new('RGB', (256, 256), color=(128, 128, 128))
            video_path = None
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, video_path

def main():
    use = '24hour'
    video_df = pl.read_parquet(
        f'./data/topic_model_videos_toponymy_{use}/video_embeddings.parquet.zstd',
        columns=['id', 'desc', 'locationCreated', 'createTime', 'playCount', 'video_path']
    )

    finetune_path = './model.safetensors'
    checkpoint_path = './models/aigc/checkpoints/lora_checkpoint_epoch_8'

    # Load processor and baseline model
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/swinv2-small-patch4-window16-256", use_fast=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swinv2-small-patch4-window16-256",
    )
   
    model.num_labels = 2
    model.config.num_labels = 2
    model.classifier = torch.nn.Linear(model.swinv2.num_features, model.num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load safetensor checkpoint if provided
    if finetune_path.endswith(".safetensors"):
        # Load safetensors checkpoint on CPU, then move to target device
        state = safe_load_file(finetune_path, device='cpu')
    else:
        # Expect a directory containing pytorch_model.bin
        state = torch.load(os.path.join(finetune_path, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, False)
    
    model = peft.PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    
    model.to(device)
    model.eval()

    # ========== Prediction over an unlabeled folder ==========
    dataset = ImageDataset(video_df, processor)

    # DataLoader
    num_workers = 4
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch], dim=0),
            [item[1] for item in batch],
        ),
    )

    # Run prediction and collect results
    threshold = 0.5
    rows = []
    with torch.no_grad():
        for pixel_values, paths in tqdm(dataloader, desc="Predict"):
            pixel_values = pixel_values.to(device)
            logits = model(pixel_values).logits
            probs_fake = logits.softmax(dim=-1)[:, 1].cpu().numpy()
            preds = (probs_fake > threshold).astype(int)
            for pth, prob, pred in zip(paths, probs_fake, preds):
                rows.append({"path": pth, "prob_fake": prob, "pred": int(pred)})

    # Save to Parquet
    save_path = f'./data/aigc/{use}_{threshold}_finetune_predictions.parquet.zstd'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pl.from_dicts(rows).write_parquet(save_path, compression='zstd')
    print(f"Saved predictions for {len(rows)} images to {save_path}")

    # print number of images considered fake
    num_fake = sum(1 for row in rows if row["pred"] == 1)
    print(f"Number of images considered fake: {num_fake} out of {len(rows)} total images.")

if __name__ == "__main__":
    main()