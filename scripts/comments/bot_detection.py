import datetime
import os
from typing import List, Tuple, Dict

import hdbscan
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

class EngagementTimeSeries(Dataset):
    def __init__(self, content_data: pl.DataFrame, ref_time: datetime):
        """
        Initialize the dataset with content engagement data
        
        Args:
            content_data: DataFrame with user_id, content_time, engage_time columns
            ref_time: Reference timestamp for the analysis window
        """
        self.ref_time = ref_time
        self.user_series = self._prepare_time_series(content_data)
        self.compressed_series = self._compress_series()

    def _prepare_time_series(self, content_data: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare time series data using Polars operations
        Returns DataFrame with user_id and time differences
        """
        # Calculate time differences from reference time
        prepared_data = content_data.with_columns([
            (pl.col('content_time') - self.ref_time).dt.total_seconds().alias('orig_diff'),
            (pl.col('engage_time') - self.ref_time).dt.total_seconds().alias('engage_diff')
        ])
        
        # Sort by user and engagement time
        prepared_data = prepared_data.sort(['user_id', 'engage_diff'])
        
        return prepared_data

    def _compress_series(self) -> List[torch.Tensor]:
        """
        Apply modified RLE compression using fully vectorized operations.
        Returns a list of compressed time series tensors.
        """
        # Calculate time differences within each user group
        compressed = self.user_series.with_columns([
            pl.col('engage_diff').diff().over('user_id').alias('time_diff')
        ])
        
        compressed = compressed.with_columns(
            pl.when(pl.col('time_diff') > 1)\
            .then(-1 * (pl.col('time_diff') - 1))\
            .otherwise(None)\
            .alias('gap')
        )
        
        # Convert to tensor format
        compressed_series = []
        user_groups = compressed.partition_by('user_id', maintain_order=True)
        
        for group in user_groups:
            series = []
            values = group.select(['orig_diff', 'gap']).rows()
            
            for orig, gap in values:
                if gap is not None:
                    series.append(gap)
                series.append(orig)
            
            if series:
                compressed_series.append(torch.tensor(series, dtype=torch.float32))
        
        return compressed_series

    def __len__(self) -> int:
        return len(self.compressed_series)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.compressed_series[idx]

class LSTMVariationalAutoencoder(nn.Module):
    def __init__(self, 
                 hidden_size: int = 64,
                 latent_size: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize the LSTM-based Variational Autoencoder
        
        Args:
            hidden_size: Number of features in the hidden state
            latent_size: Size of the latent representation
            num_layers: Number of recurrent layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,  # Single feature (timestamp)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # VAE components
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input sequence into latent space"""
        # Run LSTM encoder
        _, (hidden, _) = self.encoder_lstm(x.unsqueeze(-1))
        
        # Use last layer's hidden state
        hidden = hidden[-1]
        
        # Get mean and variance
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode the latent representation"""
        # Repeat latent vector for sequence length
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Run LSTM decoder
        output, _ = self.decoder_lstm(z_repeated)
        
        # Generate final output
        return self.output_layer(output)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, x.size(1))
        return reconstruction, mu, log_var

class RTbust:
    def __init__(self,
                 hidden_size: int = 64,
                 latent_size: int = 32,
                 num_layers: int = 2,
                 min_cluster_size: int = 5,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 100):
        """
        Initialize RTbust bot detection system
        
        Args:
            hidden_size: Hidden size for LSTM
            latent_size: Size of latent representation
            num_layers: Number of LSTM layers
            min_cluster_size: Minimum size for HDBSCAN clusters
            learning_rate: Learning rate for VAE training
            num_epochs: Number of training epochs
        """
        self.model = LSTMVariationalAutoencoder(
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers
        )
        self.min_cluster_size = min_cluster_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
    def train(self, dataset: EngagementTimeSeries):
        """Train the VAE model"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create DataLoader with padding
        def collate_fn(batch):
            # Pad sequences to max length in batch
            max_len = max(x.size(0) for x in batch)
            padded = torch.zeros(len(batch), max_len)
            for i, x in enumerate(batch):
                padded[i, :x.size(0)] = x
            return padded
            
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstruction, mu, log_var = self.model(batch)
                
                # Compute loss
                recon_loss = nn.MSELoss()(reconstruction, batch.unsqueeze(-1))
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def detect_bots(self, dataset: EngagementTimeSeries) -> Dict[str, bool]:
        """
        Detect bots using the trained model
        
        Returns:
            Dictionary mapping user_ids to boolean (True if bot)
        """
        self.model.eval()
        
        # Extract features for all users
        features = []
        with torch.no_grad():
            for series in dataset.compressed_series:
                mu, _ = self.model.encode(series.unsqueeze(0))
                features.append(mu.numpy())
        
        features = np.vstack(features)
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1
        )
        cluster_labels = clusterer.fit_predict(features)
        
        # Create results dictionary
        results = {}
        for user_id, label in zip(dataset.user_series.keys(), cluster_labels):
            # Label as bot if assigned to any cluster (-1 is noise)
            results[user_id] = (label != -1)
            
        return results

def main():
    video_df = None
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours', '19')
    video_pbar = tqdm.tqdm(total=60*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                if video_df is None:
                    video_df = batch_video_df
                else:
                    video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')

    comments_dir_path = os.path.join('.', 'data', 'comments')
    comment_df = None
    comment_pbar = tqdm.tqdm(total=len(list(os.listdir(comments_dir_path))), desc='Reading comments')
    for dir_name in os.listdir(comments_dir_path):
        comment_pbar.update(1)
        if dir_name.endswith('.zip'):
            continue
        comment_path = os.path.join(comments_dir_path, dir_name, 'comments.parquet.zstd')
        if not os.path.exists(comment_path):
            continue
        file_df = pl.read_parquet(comment_path)
        if comment_df is None:
            comment_df = file_df
        else:
            comment_df = pl.concat([comment_df, file_df], how='diagonal_relaxed')

    video_df = video_df.select(['video_id', 'createTime']).rename({'createTime': 'video_create_time'})
    comment_df = comment_df.rename({'create_time': 'comment_create_time'})
    comment_df = comment_df.join(video_df, left_on='aweme_id', right_on='video_id', how='left')
    comment_df = comment_df.filter(pl.col('video_create_time').is_not_null() & pl.col('comment_create_time').is_not_null())
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('comment_create_time')).alias('comment_create_time_epoch'))
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('video_create_time')).alias('video_create_time_epoch'))
    comment_df = comment_df.with_columns((pl.col('comment_create_time_epoch') - pl.col('video_create_time_epoch')).alias('time_offset'))
    comment_df = comment_df.with_columns(pl.col('time_offset').dt.total_seconds().alias('time_offset_seconds'))
    engagement_data = comment_df.rename({'uid': 'user_id', 'video_create_time_epoch': 'content_time', 'comment_create_time_epoch': 'engage_time', 'aweme_id': 'content_id'})\
        .select(['user_id', 'content_id', 'content_time', 'engage_time'])
    
    # Initialize with reference time
    ref_time = comment_df['video_create_time_epoch'].min()
    dataset = EngagementTimeSeries(engagement_data, ref_time)
    
    # Create and train RTbust
    rtbust = RTbust()
    rtbust.train(dataset)
    
    # Detect bots
    results = rtbust.detect_bots(dataset)
    
    # Print results
    for user_id, is_bot in results.items():
        print(f"User {user_id}: {'Bot' if is_bot else 'Human'}")


if __name__ == "__main__":
    main()