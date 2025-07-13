
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SimpleTextDataset(Dataset):
    def __init__(self, tokenizer, split="train", block_size=128):
        with open(f"data/{split}.txt", 'r', encoding='utf-8') as f:
            data = f.read()
        tokens = tokenizer.encode(data).ids
        self.inputs = []
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = tokens[i:i+block_size]
            self.inputs.append((chunk[:-1], chunk[1:]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x, y = self.inputs[idx]
        return {
            'input_ids': torch.tensor(x, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.long)
        }

def get_dataloader(tokenizer, split="train", batch_size=8):
    ds = SimpleTextDataset(tokenizer, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
