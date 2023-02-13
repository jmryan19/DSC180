from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
DATA_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/bnpp_224_pandas/'

class PreprocessedImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        filepath = row[2]  
        val = row[0]
        heart = row[1]
        im = torch.load(DATA_DIR_PATH + filepath)
        return im.view(1, 224, 224).expand(3, -1, -1), val, heart