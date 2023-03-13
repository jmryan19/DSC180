import torch
import h5py
import numpy as np
import numpy as np
import pandas as pd
import h5py
import torch
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
import os

PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_images_fixed.hdf5'
DF_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/data_fixed.csv'
DATA_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/bnpp_224_pandas/'
TEST_PATH = '/home/jmryan/private/DSC180/A/test/testdata.csv'
TRAIN_PATH = '/home/jmryan/private/DSC180/A/train/traindata.csv'
VAL_PATH = '/home/jmryan/private/DSC180/A/val/valdata.csv'
SAVE_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_256_pandas/'
LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_lung_224_pandas/'
HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_heart_224_pandas/'

SIZE = [224,224]


def read_ins():
    test = pd.read_csv(TEST_PATH)
    val = pd.read_csv(VAL_PATH)
    train = pd.read_csv(TRAIN_PATH)
    full = pd.concat([test,val,train])
    full['id'] = full['filepaths'].apply(lambda x: x.split('/')[0])
    
    df = pd.read_csv(DF_PATH)
    df = df.reset_index()
    df.columns = ['index','id'] + list(df.columns)[2:]
    merged = df.merge(full, how='inner', on='id').set_index('index')
    
    h5 = h5py.File(PATH, 'r')
    im = h5['training_images']
    return merged, im


def change_im(im):
    pil = T.ToPILImage()
    tens = T.ToTensor()
    resize = T.Resize(SIZE, interpolation=T.InterpolationMode.BILINEAR)
    im = tens(resize(pil(im)))[0]
    return im

def fix_segments(im, lung = True):
    if lung:
        segment = np.add(im[:,:,0], im[:,:,1]) > im[:,:,1].mean()
    else:
        segment = im[:,:,2] > im[:,:,2].mean()
    return torch.tensor(segment.astype(float))

def save_files():
    merged, im = read_ins()
    keys = []
    file_paths = []
    i = 0
    for i in range(len(im)):
        row = merged.iloc[i]
        key = row.id
        full_im = torch.load(DATA_DIR_PATH + row.filepaths)
        #seg_im = change_im(full_im) * change_im(fix_segments(im[i]))
        lung_seg = change_im(full_im) * change_im(fix_segments(im[i]))
        heart_seg = change_im(full_im) * change_im(fix_segments(im[i], lung = False))
        
        lung_folder_path = LUNG_PATH + str(key) + '/'
        heart_folder_path = HEART_PATH + str(key) + '/'
            
        if not os.path.exists(lung_folder_path):
            os.mkdir(lung_folder_path)
            
        if not os.path.exists(heart_folder_path):
            os.mkdir(heart_folder_path)
            
        lung_file_path = lung_folder_path + f'{key}_224.pandas'
        heart_file_path = heart_folder_path + f'{key}_224.pandas'

        torch.save(lung_seg, lung_file_path)
        torch.save(heart_seg, heart_file_path)
        #file_paths.append(file_path)
   
        if i % 500 == 0:
                print(i)
            
#     merged['seg_paths'] = file_paths
#     merged.to_csv('/home/jmryan/teams/dsc-180a---a14-[88137]/segmented_256_datapaths_meta.csv')
                        
save_files()
   