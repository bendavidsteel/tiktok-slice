import os

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import tqdm

from transformers import GPT2Config, GPT2LMHeadModel

from result_funcs import get_result_paths

class IDDataset(Dataset):
    def __init__(self, ids, sequence_length=10):
        self.ids = [self.binary_to_tensor(id) for id in ids]
        self.seq_len = sequence_length

    def binary_to_tensor(self, binary_id):
        return torch.tensor([int(bit) for bit in binary_id], dtype=torch.float32)

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, index):
        return (
            torch.stack(self.ids[index:index+self.seq_len]), 
            torch.stack(self.ids[index+1:index+self.seq_len+1])
        )

def train(model, loader, epochs=5):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        current_loss = 0
        for input_ids, labels in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}, Loss: {current_loss}"):
            outputs = model(inputs_embeds=input_ids)
            loss = loss_fn(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

def predict_next_id(model, input_id):
    model.eval()
    with torch.no_grad():
        input_tensor = dataset.binary_to_tensor(input_id).unsqueeze(0)
        output = model.generate(input_tensor, max_length=dataset.seq_len + 1)
        predicted_id = ''.join(str(int(x)) for x in output[0][-1])
        return predicted_id

def get_video_df(result_path):
    result_df = pd.read_parquet(result_path, columns=['result', 'args'])
    result_df['return'] = result_df['result'].map(lambda r: r['return'])
    result_df['id'] = result_df['return'].map(lambda r: r['id'] if r and 'id' in r else None)
    video_df = result_df[result_df['id'].map(lambda i: i is not None)]
    
    return video_df

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    result_paths = list(get_result_paths(data_dir_path))
    result_paths = sorted(result_paths)

    df = None
    result_paths = result_paths[:10]
    for result_path in tqdm.tqdm(result_paths):
        batch_df = get_video_df(result_path)

        if df is None:
            df = batch_df
        else:
            df = pd.concat([df, batch_df])

    df['bits'] = df['id'].map(lambda i: np.array([int(b) for b in format(int(i), '064b')]))

    # Example IDs
    ids = df['bits'].tolist()
    seq_len = 2048
    dataset = IDDataset(ids, sequence_length=seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Assuming each ID has 64 bits
    config = GPT2Config(
        vocab_size=64,  # Binary 0 or 1
        n_positions=seq_len,  # Maximum sequence length
        n_embd=64,  # Embedding size
        n_layer=2,  # Number of transformer layers
        n_head=2  # Number of attention heads
    )
    model = GPT2LMHeadModel(config)

    train(model, loader, epochs=5)

    next_id = predict_next_id(model, "010101...")
    print("Predicted next ID:", next_id)

if __name__ == "__main__":
    main()
