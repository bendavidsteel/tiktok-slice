import datetime
import os
import pathlib
from typing import Optional, Dict, Any, List

import av
import hydra
from PIL import Image
import polars as pl
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file as safe_load_file
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def get_frames_from_video(video_path, num_frames=5):
    """Extract multiple frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of PIL Images
    """
    frames = []
    
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError:
        return frames
    
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    
    # If total_frames is not available, estimate from duration
    if total_frames == 0:
        duration = container.duration
        if duration:
            fps = video_stream.average_rate
            if fps:
                total_frames = int((duration / av.time_base) * float(fps))
    
    if total_frames == 0:
        # Fallback: decode all frames and count
        frame_list = []
        for frame in container.decode(video=0):
            frame_list.append(frame)
        total_frames = len(frame_list)
        
        if total_frames > 0:
            # Sample frames from the decoded list
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for idx in indices:
                image = frame_list[idx].to_image().convert('RGB')
                frames.append(image)
    else:
        # Calculate frame indices to sample
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for target_idx in indices:
            # Seek to target frame
            timestamp = int(target_idx * video_stream.time_base * video_stream.average_rate.denominator / video_stream.average_rate.numerator)
            container.seek(timestamp, stream=video_stream)
            
            # Decode frame
            for frame in container.decode(video=0):
                image = frame.to_image().convert('RGB')
                frames.append(image)
                break
    
    container.close()
    return frames


def get_middle_frame(video_path):
    """Extract middle frame from video (fallback for single frame)."""
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError:
        return None
    
    video_stream = container.streams.video[0]
    duration = container.duration
    
    if duration:
        middle_time_seconds = (duration / av.time_base) / 2
        middle_timestamp = int(middle_time_seconds / video_stream.time_base)
        container.seek(middle_timestamp, stream=video_stream)
    
    for frame in container.decode(video=0):
        image = frame.to_image()
        break
    else:
        return None
    
    container.close()
    image = image.convert('RGB')
    return image


class LabeledVideoDataset(Dataset):
    """Dataset for labeled videos/images with multi-frame sampling."""
    def __init__(self, data_df, processor, label_col='label', path_col='video_path', 
                 frames_per_video=5, use_multi_frame=True):
        """
        Args:
            data_df: DataFrame with columns for paths and labels
            processor: Image processor
            label_col: Name of label column
            path_col: Name of path column
            frames_per_video: Number of frames to sample per video
            use_multi_frame: Whether to use multi-frame sampling
        """
        self.processor = processor
        self.data_df = data_df
        self.label_col = label_col
        self.path_col = path_col
        self.frames_per_video = frames_per_video
        self.use_multi_frame = use_multi_frame
        
        # Build an expanded index mapping
        self.samples = []
        for idx in range(len(data_df)):
            video_path = data_df[path_col][idx]
            label = data_df[label_col][idx]
            
            if self.use_multi_frame and video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # For videos, create multiple samples (one per frame)
                for frame_idx in range(frames_per_video):
                    self.samples.append({
                        'video_idx': idx,
                        'frame_idx': frame_idx,
                        'path': video_path,
                        'label': label
                    })
            else:
                # For images or when not using multi-frame, single sample
                self.samples.append({
                    'video_idx': idx,
                    'frame_idx': 0,
                    'path': video_path,
                    'label': label
                })
        
        # Cache for loaded frames to avoid reloading videos
        self.frame_cache = {}
        
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path):
        """Load and cache video frames."""
        if video_path not in self.frame_cache:
            if self.use_multi_frame:
                frames = get_frames_from_video(video_path, self.frames_per_video)
                if not frames:
                    # Fallback to single middle frame
                    middle_frame = get_middle_frame(video_path)
                    if middle_frame:
                        frames = [middle_frame] * self.frames_per_video
                    else:
                        # Create black frames as fallback
                        frames = [Image.new('RGB', (256, 256), color='black')] * self.frames_per_video
            else:
                middle_frame = get_middle_frame(video_path)
                if middle_frame:
                    frames = [middle_frame]
                else:
                    frames = [Image.new('RGB', (256, 256), color='black')]
            
            self.frame_cache[video_path] = frames
        
        return self.frame_cache[video_path]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        frame_idx = sample['frame_idx']
        
        # Get frame(s) from video or load image
        if path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            frames = self._load_video_frames(path)
            # Get the specific frame for this sample
            image = frames[min(frame_idx, len(frames) - 1)]
        else:
            # Load image directly
            image = Image.open(path).convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)
    
    def clear_cache(self):
        """Clear the frame cache to free memory."""
        self.frame_cache.clear()


class Trainer:
    """Training class for LoRA fine-tuning."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        num_epochs=10,
        checkpoint_dir='./checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs * len(train_loader)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (pixel_values, labels) in enumerate(progress_bar):
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values)
            loss = self.criterion(outputs.logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for pixel_values, labels in tqdm(self.val_loader, desc="Validation"):
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values)
                loss = self.criterion(outputs.logits, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self):
        """Full training loop."""
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics)
                print(f"New best model saved! Val loss: {best_loss:.4f}")

            # Clear cache after each epoch to free memory
            if hasattr(self.train_loader.dataset, 'clear_cache'):
                self.train_loader.dataset.clear_cache()
            if hasattr(self.val_loader.dataset, 'clear_cache'):
                self.val_loader.dataset.clear_cache()
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'lora_checkpoint_epoch_{epoch+1}'
        )

        self.model.save_pretrained(checkpoint_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch + 1,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        import json
        with open(checkpoint_path.replace(checkpoint_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(hydra_config):
    # Configuration
    config = hydra_config['finetune']
    
    # Add multi-frame sampling configuration
    frames_per_video = config.get('frames_per_video', 5)  # Default to 5 frames per video
    use_multi_frame = config.get('use_multi_frame', True)  # Enable by default
    
    # Load labeled data
    print("Loading labeled data...")
    df = pl.read_excel('./data/annotations-tk-20250626-from-results.xlsx')
    filenames = os.listdir('./data/aif_aigc')
    filename_ids = [f.split('.')[0] for f in filenames if f.endswith('.mp4')]
    df = df.filter(pl.col('id').is_in(filename_ids))
    df = df.with_columns(pl.format("./data/aif_aigc/{}.mp4", pl.col('id')).alias('video_path'))
    df = df.filter(pl.col('choice').is_in(['GenAI', 'Not GenAI']))
    df = df.with_columns((pl.col('choice') == 'GenAI').alias('label'))

    # Initialize processor and model
    print("Initializing model and processor...")
    processor = AutoImageProcessor.from_pretrained(config['base_model'], use_fast=True)
    
    model = AutoModelForImageClassification.from_pretrained(
        config['base_model'],
        num_labels=config['num_labels'],
        ignore_mismatched_sizes=True
    )

    print("Loading fine-tuned model checkpoint...")
    ckpt_path = config['checkpoint_path']
    # Load safetensor checkpoint if provided
    if ckpt_path.endswith(".safetensors"):
        # Load safetensors checkpoint on CPU, then move to target device
        state = safe_load_file(ckpt_path, device='cpu')
    else:
        # Expect a directory containing pytorch_model.bin
        state = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location='cpu')
    model.load_state_dict(state, False)
    
    # Configure LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=list(config['target_modules']),
        lora_dropout=config['lora_dropout'],
        bias="none",
        modules_to_save=["classifier"]  # Save only classifier parameters
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create datasets with multi-frame sampling
    print(f"Creating datasets with {frames_per_video} frames per video...")
    full_dataset = LabeledVideoDataset(
        df,
        processor,
        label_col='label',
        path_col='video_path',
        frames_per_video=frames_per_video,
        use_multi_frame=use_multi_frame
    )
    
    # Calculate effective dataset size
    num_videos = len(df)
    effective_samples = len(full_dataset)
    print(f"Original videos: {num_videos}")
    print(f"Effective training samples after frame sampling: {effective_samples}")
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Data augmentation factor: {effective_samples / num_videos:.1f}x")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    
    # Optional: Run inference with the fine-tuned model
    print("\nYou can now load the fine-tuned model for inference using:")
    print("model = AutoModelForImageClassification.from_pretrained(base_model)")
    print("model = PeftModel.from_pretrained(model, 'path/to/lora/checkpoint')")


if __name__ == "__main__":
    main()