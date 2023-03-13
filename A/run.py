import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os, sys
import math
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath('.')))
from sklearn.model_selection import train_test_split
from helpers.supernet import SuperNet 
from helpers.gradcam import NetworkGradCAM
from helpers.xrai import XRai
from helpers.image_dataset import ImageDataset
from helpers.classynet import ClassyNet
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from helpers.lightning_interface import *
import matplotlib as mpl
from skimage.transform import resize
import os
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms


FULL_HSIAO_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/segmented_datapaths_meta.csv'
HSIAO_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/bnpp_224_pandas/'
HSIAO_LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_lung_224_pandas/'
HSIAO_HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/seg_heart_224_pandas/'

FULL_MIMIC_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/final_mimic_paths.csv'
MIMIC_DIR_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_224_pandas/'
MIMIC_LUNG_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_lung_224_pandas/'
MIMIC_HEART_PATH = '/home/jmryan/teams/dsc-180a---a14-[88137]/mimic_seg_heart_224_pandas/'
final_lin = [[2048, 4096], [4096, 2]]

BATCH_SIZE = 32

def create_infra():
    # LOAD DF W DATAPATHS
    seg = pd.read_csv(FULL_HSIAO_PATH, index_col = 0)
    mim = pd.read_csv(FULL_MIMIC_PATH, index_col=0)
    seg['key'] = seg.filepaths.apply(lambda x: x.split('/')[0])
    
    # CREATE TRAIN, VAL, TEST
    hsiao_train, hsiao_val = train_test_split(seg[['heart', 'id']].to_numpy(), test_size=0.2, random_state=42)
    mim_train, temp = train_test_split(mim.to_numpy(), test_size=0.2, random_state=42)
    mim_val, mim_test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # CREATE IMAGEDATASETS
    hsiao_full_train_dataset = ImageDataset(hsiao_train, mimic=False, seg=False)
    hsiao_full_val_dataset = ImageDataset(hsiao_val, mimic=False, seg=False)

    mim_full_train_dataset = ImageDataset(mim_train, mimic=True, seg=False)
    mim_full_val_dataset = ImageDataset(mim_val, mimic=True, seg=False)
    mim_full_test_dataset = ImageDataset(mim_test, mimic=True, seg=False)

    hsiao_seg_train_dataset = ImageDataset(hsiao_train, mimic=False, seg=True)
    hsiao_seg_val_dataset = ImageDataset(hsiao_val, mimic=False, seg=True)

    mim_seg_train_dataset = ImageDataset(mim_train, mimic=True, seg=True)
    mim_seg_val_dataset = ImageDataset(mim_val, mimic=True, seg=True)
    mim_seg_test_dataset = ImageDataset(mim_test, mimic=True, seg=True)
    
    # CREATE DATALOADERS FOR TRAINING
    hsiao_full_train_dl = DataLoader(hsiao_full_train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    hsiao_full_val_dl = DataLoader(hsiao_full_val_dataset, batch_size=BATCH_SIZE, num_workers = 16, shuffle=False)

    hsiao_seg_train_dl = DataLoader(hsiao_seg_train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    hsiao_seg_val_dl = DataLoader(hsiao_seg_val_dataset, batch_size=BATCH_SIZE, num_workers = 16, shuffle=False)

    mim_full_train_dl = DataLoader(mim_full_train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    mim_full_val_dl = DataLoader(mim_full_val_dataset, batch_size=BATCH_SIZE, num_workers = 16, shuffle=False)

    mim_seg_train_dl = DataLoader(mim_seg_train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    mim_seg_val_dl = DataLoader(mim_seg_val_dataset, batch_size=BATCH_SIZE, num_workers = 16, shuffle=False)
    
    # LOAD IN RESNET WITH PRETRAINED WEIGHTS
    model = resnet152(weights=ResNet152_Weights.DEFAULT)

    return [[hsiao_full_train_dl, hsiao_full_val_dl], [mim_full_train_dl, mim_full_val_dl],
              [hsiao_seg_train_dl, hsiao_seg_val_dl], [mim_seg_train_dl, mim_seg_val_dl]], model, [mim_full_test_dataset, mim_seg_test_dataset]

def train_all():
    train_vals, model, tests = create_infra()
    
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    
    early_stop_callback = EarlyStopping(monitor="val_auc", min_delta=0.01, patience=8, verbose=False, mode="max",stopping_threshold = 0.9)
    
    titles = ['hsiao_base_full_present', 'mim_base_full_present', 'hsiao_base_seg_present', 'mim_base_seg_present']
    
    # TRAINING LOOP
    for i in range(4):
        print(titles[i])
        net = ClassyNet(title= titles[i], layer_defs=None, linear_layers = final_lin, is_transfer=True, 
           model = model, lr_scheduler=True, lr = 1e-5, print_on = True)
        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=35, 
            enable_progress_bar=False,
            check_val_every_n_epoch = 1,
            callbacks=[early_stop_callback],
            logger=False,
            enable_checkpointing=False)
        net.train()
        trainer.fit(net, train_vals[i][0], train_vals[i][1])
        torch.save(net.state_dict(), f'models/{titles[i]}')
    
def test_all():
    #GENERATE PREDICTIONS FOR EACH MODEL
    train_vals, model, [mim_full_test_dataset, mim_seg_test_dataset] = create_infra()
    final_model_paths = ['hsiao_base_full_9_reg', 'mim_base_seg_5_reg','hsiao_base_seg_5_reg', 'mim_base_full_5_reg']
    for each in final_model_paths:
        title = each + '_eval_present'
        print(title)
        net = ClassyNet(title= title, layer_defs=None, linear_layers = final_lin, is_transfer=True, 
           model = model, lr_scheduler=True, lr = 1e-5, print_on = True)
        net.to('cuda')
        net.load_state_dict(torch.load(f'models/{each}'))
        net.eval()
        if 'seg' in title:
            test = mim_seg_test_dataset
        else:
            test = mim_full_test_dataset
        probs = []
        grounds = []
        for i in range(len(test)):
            grounds.append(test[i][1])
            probs.append(F.softmax(net(test[i][0].view((1,3,224,224)).to('cuda')).detach())[:,1].item())
            if i % 100 == 0:
                print(i)
        np.array(probs).tofile(f'{title}.csv', sep = ',')
        np.array(grounds).tofile(f'{each}_grounds.csv', sep = ',')
        
    #LOAD IN PREDICTIONS JUST GENERATED
    
    hsiao_full = np.loadtxt('models/' + final_model_paths[0] + '_eval_present.csv',
                 delimiter=",", dtype=float)
    mim_seg = np.loadtxt('models/' + final_model_paths[1] + '_eval_present.csv',
                     delimiter=",", dtype=float)
    hsiao_seg = np.loadtxt('models/' + final_model_paths[2] + '_eval_present.csv',
                     delimiter=",", dtype=float)
    mim_full = np.loadtxt('models/' + final_model_paths[3] + '_eval_present.csv',
                     delimiter=",", dtype=float)
    full_ground = np.loadtxt('models/' + final_model_paths[3] + '_grounds_present.csv',
                     delimiter=",", dtype=float)

    seg_ground = np.loadtxt('models/' + final_model_paths[2] + '_grounds_present.csv',
                     delimiter=",", dtype=float)
    
    #CALCULATE TEST AUCS, PRCS, ACCURACIES ON ALL MODELS
    
    hsiao_full_auc = roc_auc_score(full_ground, hsiao_full)
    hsiao_full_prc = precision_score(full_ground, (hsiao_full > 0.5).astype(int), zero_division=0)
    hsiao_full_acc = accuracy_score(full_ground, (hsiao_full > 0.5).astype(int))

    mim_seg_auc = roc_auc_score(seg_ground, mim_seg)
    mim_seg_prc = precision_score(seg_ground, (mim_seg > 0.5).astype(int), zero_division=0)
    mim_seg_acc = accuracy_score(seg_ground, (mim_seg > 0.5).astype(int))

    hsiao_seg_auc = roc_auc_score(seg_ground, hsiao_seg)
    hsiao_seg_prc = precision_score(seg_ground, (hsiao_seg > 0.5).astype(int), zero_division=0)
    hsiao_seg_acc = accuracy_score(seg_ground, (hsiao_seg > 0.5).astype(int))

    mim_full_auc = roc_auc_score(full_ground, mim_full)
    mim_full_prc = precision_score(full_ground, (mim_full > 0.5).astype(int), zero_division=0)
    mim_full_acc = accuracy_score(full_ground, (mim_full > 0.5).astype(int))
    
    #PLOT ALL TEST AUCS AND PRCS
    verc = [[full_ground, hsiao_full, f'UCSD Full, AUROC of {np.round(hsiao_full_auc, 3)}', f'UCSD Full, AUPRC of {np.round(hsiao_full_prc, 3)}'], 
            [full_ground, mim_full, f'MIMIC Full, AUROC of {np.round(mim_full_auc, 3)}', f'MIMIC Full, AUPRC of {np.round(mim_full_prc, 3)}'], 
            [seg_ground, hsiao_seg, f'UCSD Segmented, AUROC of {np.round(hsiao_seg_auc, 3)}', f'UCSD Segmented, AUPRC of {np.round(hsiao_seg_prc, 3)}'], 
            [seg_ground, mim_seg, f'MIMIC Segmented, AUROC of {np.round(mim_seg_auc, 3)}', f'MIMIC Segmented, AUPRC of {np.round(mim_seg_prc, 3)}']]
    rocs = []
    prcs = []
    for ver in verc:
        fpr, tpr, thresholds = roc_curve(ver[0], ver[1])
        precision, recall, thresholds = precision_recall_curve(ver[0], ver[1])
        roc = pd.DataFrame([fpr,tpr]).T
        roc.columns = ['False Positive Rate', 'True Positive Rate']
        roc['Model'] = [ver[2]] * len(roc)
        prc = pd.DataFrame([precision, recall]).T
        prc.columns = ['Precision', 'Recall']
        prc['Model'] = [ver[3]] * len(prc)
        rocs.append(roc)
        prcs.append(prc)
    all_rocs = pd.concat(rocs)
    all_prcs = pd.concat(prcs)
    
    plt.figure(figsize=(10,8))
    sns.lineplot(data=all_rocs, x='False Positive Rate', y='True Positive Rate', hue='Model').set(title = 'All Models ROC')
    plt.savefig('final_figs/all_roc_present.png', pad_inches=0.2, bbox_inches='tight')
    
    plt.figure(figsize=(10,8))
    sns.lineplot(data = all_prcs, x='Precision', y='Recall', hue='Model').set(title='All Models PRC')
    plt.savefig('final_figs/all_prc_present.png', pad_inches=0.2, bbox_inches='tight')
    

def main():
    args = sys.argv[1:]
    if args[0] == 'test':
        test_all()
    elif args[0] == 'train':
        train_all()
    
if __name__ == '__main__':
    main()