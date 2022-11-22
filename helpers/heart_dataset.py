from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class PreprocessedImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        filepath = row[7]  
        val = row[0]
        heart = row[6]
        im = torch.load(filepath)
        return im.view(1, 224, 224).expand(3, -1, -1), val, heart