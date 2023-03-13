import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

FULL_HSIAO_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/segmented_datapaths_meta.csv'
HSIAO_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/bnpp_224_pandas/'
HSIAO_LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_lung_224_pandas/'
HSIAO_HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_heart_224_pandas/'

FULL_MIMIC_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/final_mimic_paths.csv'
MIMIC_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_224_pandas/'
MIMIC_LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_lung_224_pandas/'
MIMIC_HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_heart_224_pandas/'

class ImageDataset(Dataset):
    def __init__(self, df, mimic, transform=None, target_transform=None, seg = False):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.seg = seg
        if mimic:
            self.path = MIMIC_DIR_PATH
            if self.seg:
                self.heart = MIMIC_HEART_PATH
                self.lung = MIMIC_LUNG_PATH
        else:
            self.path = HSIAO_DIR_PATH
            if self.seg:
                self.heart = HSIAO_HEART_PATH
                self.lung = HSIAO_LUNG_PATH
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        filepath = row[1]
        
        val = row[0]
        if self.seg:
            full = torch.load(self.path + filepath + '/' + filepath + '_224.pandas')
            lung = torch.load(self.lung + filepath + '/' + filepath + '_224.pandas')
            heart = torch.load(self.heart + filepath + '/' + filepath + '_224.pandas')
            im = torch.stack([full, lung, heart])
#             print(self.path + filepath + '/' + filepath + '_224.pandas', 
#                   self.lung + filepath + '/' + filepath + '_224.pandas',
#                  self.heart + filepath + '/' + filepath + '_224.pandas')
        else:
            im = torch.load(self.path + filepath + '/' + filepath + '_224.pandas').view(1, 224, 224).expand(3, -1, -1)
        return im, int(val)