import torch
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
import h5py
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
#from tensorflow.keras.utils import HDF5Matrix
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
# print(tf.__version__)
# import datetime





NORMAL_SAVE_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_224_pandas/'
SEG_SAVE_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_224_pandas/'
MIMIC_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/MIMIC_Images/'
DF_PATH = MIMIC_PATH + 'mimic_paths_meta.csv'
MIMIC_FOLDER = MIMIC_PATH + 'training_data_big/'
LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_lung_224_pandas/'
HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_heart_224_pandas/'

MODEL = 'CXR_model.json' #Brian's Lung Model Architechture
WEIGHTS = 'CXR_weights.hdf5' #Brian's Lung Model Weights

tf_SIZE = [256,256]
save_SIZE = [224,224]

def load_lung_seg_model(model_path, weights_path):
    """ Loads the lung segmentation model built by Brian Hurt, MD, MS (AiDA Lab)

    Args:
        model_path:   path to lung segmentation model architecture
        weights_path: path to lung segmentation model weights

    Returns:
        lung segmentation model (keras Model class) with weights loaded
    """
    # load liver segmentation model
    json_file = open(model_path, 'r'); loaded_model_json = json_file.read()
    json_file.close()
    
    # load model weights
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_path) # Load model weights
    
    return model


def read_ins():
    df = pd.read_csv(DF_PATH, index_col=0)
    lung_model = load_lung_seg_model(MODEL, WEIGHTS)
    return df, lung_model

def load_im(im):
#     im = Image.open(path)
    pil = T.ToPILImage()
    tens = T.ToTensor()
    resize = T.Resize(tf_SIZE, interpolation=T.InterpolationMode.BILINEAR)
    im = tens(resize(pil(im)))[0]
    return im


def change_im(im):
    pil = T.ToPILImage()
    tens = T.ToTensor()
    resize = T.Resize(save_SIZE, interpolation=T.InterpolationMode.BILINEAR)
    im = tens(resize(pil(im)))[0]
    return im

def fix_segments(im, lung = True):
    if lung:
        segment = np.add(im[:,:,0], im[:,:,1]) > im[:,:,1].mean()
    else:
        segment = im[:,:,2] > im[:,:,2].mean()
    return torch.tensor(segment.astype(float))

def run_model():
    df, lung_model = read_ins()
    file_paths = []
    for i in range(len(df)):
        if i < 10277:
            continue

        row = df.iloc[i]
        path = row.path
        ed = row.Edema
        im_key = row.im_key[:-4]
        
        im = cv2.equalizeHist(np.uint8(255 * cv2.cvtColor(plt.imread(path), cv2.COLOR_RGBA2GRAY))) / 255

        
        im = load_im(torch.tensor(im))
        tf_im = tf.convert_to_tensor(im.numpy())
        tf_im = tf.reshape(tf_im, [1, 256,256, 1])
        out = lung_model(tf_im)[0].numpy()
        
        lung_seg = change_im(im) * change_im(fix_segments(out))
        heart_seg = change_im(im) * change_im(fix_segments(out, lung = False))
        
        im = change_im(im)
        
        normal_folder_path = NORMAL_SAVE_PATH + str(im_key) + '/'
        
        lung_folder_path = LUNG_PATH + str(im_key) + '/'
        heart_folder_path = HEART_PATH + str(im_key) + '/'
            
        if not os.path.exists(normal_folder_path):
            os.mkdir(normal_folder_path)
            
        if not os.path.exists(lung_folder_path):
            os.mkdir(lung_folder_path)
            
        if not os.path.exists(heart_folder_path):
            os.mkdir(heart_folder_path)
            
        normal_file_path = normal_folder_path + f'{im_key}_224.pandas'
        torch.save(im, normal_file_path)
        
        lung_file_path = lung_folder_path + f'{im_key}_224.pandas'
        heart_file_path = heart_folder_path + f'{im_key}_224.pandas'

        torch.save(lung_seg, lung_file_path)
        torch.save(heart_seg, heart_file_path)
        file_paths.append(str(im_key) +'/' + f'{im_key}_224.pandas')
   
        if i % 100 == 0:
            print(i)
            
    df['final_paths'] = file_paths
    df.to_csv('/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_224.csv')
                        
run_model()