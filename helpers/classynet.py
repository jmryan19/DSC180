import torch
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from helpers.lightning_interface import *
import matplotlib as mpl
from skimage.transform import resize
import os

class ClassyNet(SuperFace):
    def __init__(self, title, layer_defs, linear_layers, type_model = 'classifier', lr = 1e-3, is_transfer=False, model=None, lr_scheduler = [], batch_size =32, print_on = True):
        super().__init__(layer_defs = layer_defs, model = model, lr_scheduler = lr_scheduler, lr=lr)
        
        self.title = title
        self.model = model
        self.print = print_on
        self.linear_layers = linear_layers
        self.grad = False
        self.val_heart_true_epoch = np.array([])
        self.val_heart_hat_epoch = np.array([])
        self.train_heart_true_epoch = np.array([])
        self.train_heart_hat_epoch = np.array([])
        self.val_mae_epoch = np.array([])
        self.train_mae_epoch = np.array([])
        self.train_loss_epoch = np.array([])
        self.val_loss_epoch = np.array([])
        self.val_auc = np.array([])
        self.init_model()
        
#         self.l1_loss = nn.L1Loss()
#         self.l2_loss = nn.MSELoss()
#         self.l1_str = l1
#         self.l2_str = l2
    
        
        self.tr_fpath = self.get_savefname(dset='train')
        self.val_fpath = self.get_savefname(dset='valid')
        
        if os.path.isfile(self.tr_fpath):
            os.remove(self.tr_fpath)
            os.remove(self.val_fpath)
#             print('Experiment Title Already Exists: Choose New Title')
#             assert False
        
        self.BATCH_SIZE = batch_size
        if type_model == 'regressor':
            self.loss_func = self.mae
            
        elif type_model == 'classifier':
            self.loss_func = self.cbe
            
        else:
            print('Invalid Loss Func: Not Implemented')
            assert False
            
    def get_savefname(self, dset='train'):
        return f"epoch_logs/{self.title}_logs_{dset}.csv"
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.9) 
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience = 3, verbose=True)
        return {"optimizer": optim}# "lr_scheduler": {'scheduler': lr_sched, 'monitor': 'val_auc', 
                                    #                'interval': 'epoch'}}
    
    def init_model(self):
        layers = list(self.model.children())
        lin = layers[-1]
        layers = layers[:-1]
        temp_lin = [nn.Linear(num_in,num_out) for num_in, num_out in self.linear_layers]
        total_lin = []
        for lin in temp_lin[:-1]:
            #total_lin.append(nn.Dropout(0.75))
            total_lin.append(lin)
            total_lin.append(nn.ReLU())
            total_lin.append(nn.Dropout(0.75))
        total_lin.append(temp_lin[-1])
#         total_lin.append(nn.LogSoftmax())
        self.regresser = nn.Sequential(*total_lin)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.grad:
            represents = self.layers(x).flatten(1)
            y_hat = self.regresser(represents)
        else:
            represents = x
            for i in range(len(self.layers)):
                if True:#self.current_epoch >= 5 and (i == 8 or i==7 or i ==6 or i==5 or i==4):
                    represents = self.layers[i](represents)
                    
#                 elif self.current_epoch >= 10 and i == 7:
#                     represents = self.layers[i](represents)
                    
#                 elif self.current_epoch >= 15 and i == 6:
#                     represents = self.layers[i](represents)
                    
#                 elif self.current_epoch >= 20 and i == 5:
#                     represents = self.layers[i](represents)

                else:
                    with torch.no_grad():
                        represents = self.layers[i](represents)
            y_hat = self.regresser(represents.flatten(1))
        del x
        return y_hat
    
    
    def turn_grad(self, boo):
        self.grad = boo
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        #print(y_hat.detach().mean().item(), y.mean().item(), y_hat.detach().median().item(), y.median().item())
        del x
        
#         loss = self.loss_func(y_hat, y)
        
        loss = self.loss_func(y_hat, y)
        
        arged = F.softmax(y_hat.detach())[:,1]
        

        
        loss_dic = {'loss': loss,
                    'y_hat': arged,
                    'y_true': y
                   }
        
        return loss_dic
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        #print(y_hat.detach().mean().item(), y.mean().item(), y_hat.detach().median().item(), y.median().item())
        del x
        
#         loss = self.loss_func(y_hat, y)
        
        loss = self.loss_func(y_hat, y).detach()
        
        arged = F.softmax(y_hat.detach())[:,1]

#         arged = self.softmax_np(y_hat.cpu().detach().numpy())[:,1]
    
#         print(f"Arged: {arged}")

        
        loss_dic = {'loss': loss,
                    'y_hat': arged,
                    'y_true': y
                   }
        
        return loss_dic
    
        
    def training_step_end(self, batch_loss):
        total_loss = batch_loss['loss'].mean()
        heart_true = batch_loss['y_true'].to('cpu').numpy()
        heart_hat = batch_loss['y_hat'].to('cpu').numpy()
        
        self.train_heart_true_epoch = np.append(self.train_heart_true_epoch, heart_true)
        self.train_heart_hat_epoch = np.append(self.train_heart_hat_epoch, heart_hat)
        
        torch.cuda.empty_cache()
        out = 'training_step (pre del) mem %:', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        #print(out)        
        return total_loss
    
    def validation_step_end(self, batch_loss):
        total_loss = batch_loss['loss'].mean()
        heart_true = batch_loss['y_true'].to('cpu').numpy()
        heart_hat = batch_loss['y_hat'].to('cpu').numpy()
        
        
        self.val_heart_true_epoch = np.append(self.val_heart_true_epoch, heart_true)
        self.val_heart_hat_epoch = np.append(self.val_heart_hat_epoch, heart_hat)
        
        return total_loss
        
    def training_epoch_end(self, step_outputs):
        
        losses = [loss['loss'] for loss in step_outputs]
        
        heart_true = self.train_heart_true_epoch
        heart_hat = self.train_heart_hat_epoch
        total_loss = sum(losses)/len(losses)
        #print(heart_hat)
        auc = roc_auc_score(heart_true, heart_hat)
        prc = precision_score(heart_true, (heart_hat > 0.5).astype(int), zero_division=0)
        acc = accuracy_score(heart_true, (heart_hat > 0.5).astype(int))
        
# #         self.log('train_AUC', auc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
#         self.log('train_PRC', prc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
        self.log('loss', total_loss,
                on_step=False, on_epoch=True, prog_bar=False, batch_size=self.BATCH_SIZE)
        
        
        info_dic = {'Epoch': [self.current_epoch], 'AUC': [auc], 'PRC': [prc], 'Accuracy': [acc], 'loss':[total_loss.item()]}
        print(f"Epoch {self.current_epoch}")
        if self.print:
            print(f"\tTrain {info_dic};")
#        print(f"\tTrain loss: {total_loss.item()}; mean_mae: {epoch_mae.mean()};" + 
 #            f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")
    
        if not os.path.isfile(self.tr_fpath):
            pd.DataFrame.from_dict(info_dic).to_csv(self.tr_fpath)
            
        else:
            pd.concat([pd.read_csv(self.tr_fpath, index_col=0), pd.DataFrame.from_dict(info_dic)]).to_csv(self.tr_fpath)
            
        
        self.train_loss_epoch = np.append(self.train_loss_epoch, total_loss.item())
        
        self.train_heart_true_epoch = np.array([])
        self.train_heart_hat_epoch = np.array([])
        #sch = self.lr_schedulers()
        #print(self.lr_schedulers())

        # If the selected scheduler is a ReduceLROnPlateau scheduler.

        return None
        
    def validation_epoch_end(self, step_outputs):
        losses = step_outputs

        heart_true = self.val_heart_true_epoch
        heart_hat = self.val_heart_hat_epoch
        total_loss = sum(losses)/len(losses)
        
#         print(f'HAT: {heart_hat}')

        auc = roc_auc_score(heart_true, heart_hat)
        prc = precision_score(heart_true, (heart_hat > 0.5).astype(int), zero_division=0)
        acc = accuracy_score(heart_true, (heart_hat > 0.5).astype(int))
        self.log('val_auc', auc)

#         self.log('val_AUC', auc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
#         self.log('val_PRC', prc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
# #         self.log('val_loss', total_loss,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
        
        info_dic = {'Epoch': [self.current_epoch], 'AUC': [auc], 'PRC': [prc], 'Accuracy': [acc], 'loss':[total_loss.item()]}  
        
        if self.print:
            print(f"\tVal {info_dic}")
#         print(f"\tVal loss: {total_loss.item()}; mean_mae: {epoch_mae.mean()};" + 
#              f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")

        if not os.path.isfile(self.val_fpath):
            pd.DataFrame.from_dict(info_dic).to_csv(self.val_fpath)
            
        else:
            pd.concat([pd.read_csv(self.val_fpath, index_col=0), pd.DataFrame.from_dict(info_dic)]).to_csv(self.val_fpath)
        
        self.val_auc = np.append(self.val_auc, auc)
        self.val_heart_true_epoch = np.array([])
        self.val_heart_hat_epoch = np.array([])
        self.val_loss_epoch = np.append(self.val_loss_epoch, total_loss.item())

        
        return None
        
    def get_classbalance_weights(self, y_hat, y_true, beta=0.999):
        # Get class counts
        classes, counts = torch.unique(y_true, return_counts=True)
#         print(classes,counts)
        if 1 not in classes: # Case where there are no sick samples
            counts = torch.tensor([counts[0], 0]).type_as(y_hat)#.device)
        if 0 not in classes: # Case where there are no healthy samples
            counts = torch.tensor([0, counts[0]]).type_as(y_hat)#.device)
        # Calculate weight for each class
        beta = torch.tensor(beta).type_as(y_hat)#.device)        
        one = torch.tensor(1.).type_as(y_hat)#.device)
        weights = (one - torch.pow(beta, counts)) / (one - beta)
        return weights        
    
    def cbe(self, y_hat, y_true, beta=0.9):
        weights = self.get_classbalance_weights(y_hat, y_true, beta=beta)
#         print(weights)
#         print(torch.unique(y_hat))
        cb_ce_loss = F.cross_entropy(y_hat, y_true, weight=weights)
        return cb_ce_loss 
    
    def mae(self, y_hat, y_true):
        y_true = y_true.view(-1,1)
        return torch.abs(y_true - y_hat)
    
    def softmax_np(x, axis=1):
        return np.exp(x)/np.sum(np.exp(x), axis=axis).reshape(-1,1)