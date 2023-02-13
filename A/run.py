import torch
import math
import torch.nn as nn
import pandas as pd
import os, time, sys
import numpy as np
import pytorch_lightning as pl
import h5py
sys.path.append(os.path.dirname(os.path.realpath('.')))
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
import glob
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'svg'
pio.templates.default = 'plotly_white'
from helpers.lightning_interface import *
from helpers.heart_dataset import PreprocessedImageDataset
from helpers.supernet import SuperNet
from helpers.gradcam import NetworkGradCAM
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib as mpl
from skimage.transform import resize
from helpers.gradcam import NetworkGradCAM
import matplotlib as mpl
from skimage.transform import resize
import cv2

TEST_PATH = '/home/jmryan/private/DSC180/A/test/testdata.csv'
TRAIN_PATH = '/home/jmryan/private/DSC180/A/train/traindata.csv'
VAL_PATH = '/home/jmryan/private/DSC180/A/val/valdata.csv'
MODEL_PATH = 'final_model'

def create_infra(df_train, df_val, BATCH_SIZE):
    train_dataset = PreprocessedImageDataset(df=df_train.to_numpy())
    val_dataset = PreprocessedImageDataset(df=df_val.to_numpy())
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers = 0, shuffle=False)
    
    if torch.cuda.is_available():
        dev = 'gpu'
    else:
        dev = 'cpu'
        
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    return train_dl, val_dl, dev, model

def train_all(df_train, df_val, BATCH_SIZE):
    train_dl, val_dl, dev, model = create_infra(df_train, df_val, BATCH_SIZE)
    
    test_dataset = PreprocessedImageDataset(df=pd.read_csv(TEST_PATH, index_col=0).to_numpy())
    
    lin_choices = [[[2048,1]],
               [[2048, 1024], [1024, 512], [512,256], [256,1]],
               [[2048, 256], [256,1]],
               [[2048, 4096], [4096, 2048], [2048, 512], [512, 256], [256, 1]]]
    all_aucs = []
    all_tests = []
    all_val_loss = []
    all_train_loss = []
    for lin_lay in lin_choices:
        print(lin_lay)
        net = SuperNet(layer_defs=None, linear_layers = lin_lay, is_transfer=True, 
                   model = model, lr_scheduler=True, lr = 1e-5, print_on = False)
        trainer = pl.Trainer(
            accelerator=dev,
            max_epochs=30, 
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False)
        net.train()
        trainer.fit(net, train_dl, val_dl)
        net.eval()
        test_outcomes = []
        for each in test_dataset:
            test_outcomes.append(net(each[0].reshape((1,3,224,224))).item())
        all_aucs.append(net.val_auc)
        all_tests.append(test_outcomes)
        all_val_loss.append(net.val_loss_epoch)
        all_train_loss.append(net.train_loss_epoch)
        
    np.save('run_figs/aucs',np.array(all_aucs))
    np.save('run_figs/tests', np.array(all_tests))
    np.save('run_figs/val_loss', np.array(all_val_loss))
    np.save('run_figs/train_loss', np.array(all_train_loss))
    
    plt.plot(np.arange(len(net.val_loss_epoch) - 1),  net.val_loss_epoch[1:])
    plt.plot(np.arange(len(net.train_loss_epoch)),  net.train_loss_epoch)
    plt.legend(['Val','Train'])
    plt.show()
    
def gcam(df_train, df_val, BATCH_SIZE):
    train_dl, val_dl, dev, model = create_infra(df_train, df_val, BATCH_SIZE)
    test_dataset = PreprocessedImageDataset(df=pd.read_csv(TEST_PATH, index_col=0).to_numpy())
    
    lin_lay = [[2048, 4096], [4096, 2048], [2048, 512], [512, 256], [256, 1]]
    
    net = SuperNet(layer_defs=None, linear_layers = lin_lay, is_transfer=True, 
                   model = model, lr_scheduler=True, lr = 1e-5, print_on = False)
    
    net.load_state_dict(torch.load(MODEL_PATH))
    
    gCAM = NetworkGradCAM(net)
    gCAM.eval()
    net.turn_grad(True)
    
    FN = 186
    FP = 1364
    TN = 696
    TP = 1156
    exs = [TP, FP, FN, TN]
    order = ['TP', 'FP', 'FN', 'TN']
    
    ims = []
    base_ims = []
    for num in exs:
        val = test_dataset[num][0]
        pred = gCAM(val.view((1,3,224,224)).to('cuda'))
        pred.backward(retain_graph=True)
        gradients = gCAM.get_activations_gradient()
        activations = gCAM.get_activations(val.view((1,3,224,224)).to('cuda')).detach()
        gCAM.zero_grads()
        pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])

        for i in range(pooled_gradients.shape[0]):
            activations[:, i, ...] *= pooled_gradients[i]

        heatmap = torch.mean(activations[0], dim = 0).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        heatmap2 = cv2.resize(heatmap.numpy(), (224,224))
        heatmap2 = np.uint8(255 * heatmap2)
        t2 = val.permute(1, 2, 0).numpy()
        t3 = np.uint8(255*t2)
        heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
        base_ims.append(t3)
        superimposed_img = (heatmap2 * 0.4) + t3
        superimposed_img = np.uint8(255*(superimposed_img-superimposed_img.min())/(superimposed_img.max() - superimposed_img.min()))
        ims.append(superimposed_img)
     
    for i in range(len(ims)):
        fig, axs = plt.subplots(1, 1, figsize = (8,8))
        axs.imshow(ims[i].astype(int))
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        fig.savefig(f'run_figs/{order[i]}_gcam.png', dpi=100)

        fig, axs = plt.subplots(1, 1, figsize = (8,8))
        axs.imshow(base_ims[i].astype(int))
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        fig.savefig(f'run_figs/{order[i]}_base.png', dpi=100)

def main():
    args = sys.argv[1:]
    BATCH_SIZE = eval(args[2])
    if args[0] == 'test':
        df_train, df_val = train_test_split(pd.read_csv(TEST_PATH, index_col=0), test_size = 0.2)
    elif args[0] == 'train':
        df_train = pd.read_csv(TRAIN_PATH, index_col=0)
        df_val = pd.read_csv(VAL_PATH, index_col=0)
        
    if args[1] == 'gcam':
        gcam(df_train, df_val, BATCH_SIZE)
    elif args[1] == 'model':
        train_all(df_train, df_val, BATCH_SIZE)
    
if __name__ == '__main__':
    main()