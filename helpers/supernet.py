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
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score
from helpers.lightning_interface import *
import matplotlib as mpl
from skimage.transform import resize

class SuperNet(SuperFace):
    def __init__(self, layer_defs, linear_layers, loss_func = 'mae', lr = 1e-3, is_transfer=False, model=None, lr_scheduler = [], batch_size =32, print_on = True):
        super().__init__(layer_defs = layer_defs, model = model, lr_scheduler = lr_scheduler, lr=lr)
        
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
        self.BATCH_SIZE = batch_size
        if loss_func = 'mae':
            self.loss_func self.mae
            
        elif loss_func = 'cbe':
            self.loss_func = self.cbe
            
        else:
            print('Invalid Loss Func: Not Implemented')
            assert False
            
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr) 
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
        return {"optimizer": optim}#, "lr_scheduler": {'scheduler': lr_sched, 'monitor': 'loss', 
                                    #                 'interval': 'epoch'}}
    
    def init_model(self):
        layers = list(self.model.children())
        lin = layers[-1]
        layers = layers[:-1]
        temp_lin = [nn.Linear(num_in,num_out) for num_in, num_out in self.linear_layers]
        total_lin = []
        for lin in temp_lin:
            total_lin.append(nn.Dropout(0.75))
            total_lin.append(lin)
        self.regresser = nn.Sequential(*total_lin)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.grad:
            represents = self.layers(x).flatten(1)
            y_hat = self.regresser(represents)
        else:
            represents = x
            for i in range(len(self.layers)):
                if False:#self.current_epoch >= 5 and (i == 8 or i==7 or i ==6 or i==5 or i==4):
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
        x, y, heart = batch
        
        y_hat = self(x)
        #print(y_hat.detach().mean().item(), y.mean().item(), y_hat.detach().median().item(), y.median().item())
        del x
        
#         loss = self.loss_func(y_hat, y)
        
        loss = self.mae(y_hat, y)
        
        m_a_e = self.mae(y_hat, y).detach()
        
        y_heart = ((10**y_hat) > 400).long()
        
        loss_dic = {'loss': loss,
                    'y_hat': y_hat,
                    'y_true': y,
                    'heart_true': heart,
                    'heart_hat': y_heart,
                    'mae': m_a_e
                   }
        
        return loss_dic
    
    def validation_step(self, batch, batch_idx):
        x, y, heart = batch
        
        y_hat = self(x)
        del x
#         loss = self.loss_func(y_hat, y)
        
        m_a_e = self.mae(y_hat, y).detach()
        
        loss = self.mae(y_hat, y).detach()

        y_heart = ((10**y_hat) > 400).long()
        
        loss_dic = {'loss': loss,
                    'y_hat': y_hat,
                    'y_true': y,
                    'heart_true': heart,
                    'heart_hat': y_heart,
                    'mae': m_a_e
                   }
        
        return loss_dic
    
        
    def training_step_end(self, batch_loss):
        total_loss = batch_loss['loss'].mean()
        heart_true = batch_loss['heart_true'].to('cpu').numpy()
        heart_hat = batch_loss['heart_hat'].to('cpu').numpy()
        step_mae = batch_loss['mae'].to('cpu').numpy()
        
        self.train_heart_true_epoch = np.append(self.train_heart_true_epoch, heart_true)
        self.train_heart_hat_epoch = np.append(self.train_heart_hat_epoch, heart_hat)
        self.train_mae_epoch = np.append(self.train_mae_epoch, step_mae)
        
        torch.cuda.empty_cache()
        out = 'training_step (pre del) mem %:', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        #print(out)        
        return total_loss
    
    def validation_step_end(self, batch_loss):
        total_loss = batch_loss['loss'].mean()
        heart_true = batch_loss['heart_true'].to('cpu').numpy()
        heart_hat = batch_loss['heart_hat'].to('cpu').numpy()
        step_mae = batch_loss['mae'].to('cpu').numpy()
        
        self.val_heart_true_epoch = np.append(self.val_heart_true_epoch, heart_true)
        self.val_heart_hat_epoch = np.append(self.val_heart_hat_epoch, heart_hat)
        self.val_mae_epoch = np.append(self.val_mae_epoch, step_mae)
        
        return total_loss
        
    def training_epoch_end(self, step_outputs):
        
        losses = [loss['loss'] for loss in step_outputs]
        
        heart_true = self.train_heart_true_epoch
        heart_hat = self.train_heart_hat_epoch
        epoch_mae = self.train_mae_epoch
        total_loss = sum(losses)/len(losses)
        auc = roc_auc_score(heart_true, heart_hat)
        prc = precision_score(heart_true, heart_hat, zero_division=0)
        
# #         self.log('train_AUC', auc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
#         self.log('train_PRC', prc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
        self.log('loss', total_loss,
                on_step=False, on_epoch=True, prog_bar=False, batch_size=self.BATCH_SIZE)
        
        
        info_dic = {'AUC': auc, 'PRC': prc, 'loss':total_loss.item()}
        print(f"Epoch {self.current_epoch}")
        if self.print:
            print(f"Epoch {self.current_epoch}")
            print(f"\tTrain {info_dic}; mean_mae: {epoch_mae.mean()};" + 
                 f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")
#        print(f"\tTrain loss: {total_loss.item()}; mean_mae: {epoch_mae.mean()};" + 
 #            f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")
        
        self.train_loss_epoch = np.append(self.train_loss_epoch, total_loss.item())
        
        self.train_heart_true_epoch = np.array([])
        self.train_heart_hat_epoch = np.array([])
        self.train_mae_epoch = np.array([])
        #sch = self.lr_schedulers()
        #print(self.lr_schedulers())

        # If the selected scheduler is a ReduceLROnPlateau scheduler.

        return None
        
    def validation_epoch_end(self, step_outputs):
        losses = step_outputs

        heart_true = self.val_heart_true_epoch
        heart_hat = self.val_heart_hat_epoch
        epoch_mae = self.val_mae_epoch
        total_loss = sum(losses)/len(losses)

        auc = roc_auc_score(heart_true, heart_hat)
        prc = precision_score(heart_true, heart_hat, zero_division=0)

#         self.log('val_AUC', auc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
#         self.log('val_PRC', prc,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
# #         self.log('val_loss', total_loss,
#                 on_step=False, on_epoch=True, prog_bar=True, batch_size=self.BATCH_SIZE)
        
        
        info_dic = {'AUC': auc, 'PRC': prc,'loss': total_loss.item()}
        
        if self.print:
            print(f"\tVal {info_dic}; mean_mae: {epoch_mae.mean()};" + 
                 f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")
#         print(f"\tVal loss: {total_loss.item()}; mean_mae: {epoch_mae.mean()};" + 
#              f" mean_heart_hat: {heart_hat.mean()}; mean_heart_true: {heart_true.mean()}")
        
        self.val_auc = np.append(self.val_auc, auc)
        self.val_heart_true_epoch = np.array([])
        self.val_heart_hat_epoch = np.array([])
        self.val_mae_epoch = np.array([])
        self.val_loss_epoch = np.append(self.val_loss_epoch, total_loss.item())

        
        return None
        
#     def cbe(self, y_hat, y_true):
        
    
    def mae(self, y_hat, y_true):
        y_true = y_true.view(-1,1)
        return torch.abs(y_true - y_hat)