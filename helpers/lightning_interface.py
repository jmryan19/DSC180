import torch
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class SuperFace(pl.LightningModule):
    def __init__(self, 
                 layer_defs,
                 lr = 1e-3,
                 lr_scheduler = [],
                 lr_reg = 0,
                 model = None,
                 batch_size = 32):
        super().__init__()
        if model is not None:
            self.model = model
        self.lr = lr
        self.lr_reg = lr_reg
        self.lr_scheduler = lr_scheduler
        self.layer_defs = layer_defs
        self.batch_size = batch_size
        
        self.init_model()
        
        
    def init_model(self):
        self.layers = self.unwrap_defs(self.layer_defs)
        
    def unwrap_defs(self, layers):
        final_layers = []
        for layer in layers:
            if layer['type'] == 'linear':
                final_layers.append(nn.Linear(layer['n_in'], layer['n_out']))
                if layer['relu']:
                    final_layers.append(nn.Relu())
                
            elif layer['type'] == 'conv1d':
                final_layers.append(nn.Conv1d(layer['n_in'], layer['n_out'], kernel_size = layer['kernel_size'], stride = layer['stride'], padding = layer['padding']))
                final_layers.append(nn.Relu())

           
            elif layer['type'] == 'conv2d':
                final_layers.append(nn.Conv1d(layer['n_in'], layer['n_out'], kernel_size = layer['kernel_size'], stride = layer['stride'], padding = layer['padding']))
                final_layers.append(nn.Relu())
                
            elif layer['type'] == 'maxpool1d':
                final_layers.append(nn.MaxPool1d(layer['kernel_size'],
                                           layer['stride'],
                                           layer['padding']))
                final_layers.append(nn.Relu())
            
            elif layer['type'] == 'maxpool2d':
                final_layers.append(nn.MaxPool2d(layer['kernel_size'],
                                           layer['stride'],
                                           layer_['padding']))
                final_layers.append(nn.Relu())
                
            elif layer_def['type'] == 'dropout':
                final_layers.append(nn.Dropout(layer_def['p']))
                final_layers.append(nn.Relu())
                
            elif layer_def['type'] == 'avgpool2d':
                final_layers.append(torch.nn.AdaptiveAvgPool2d(layer_def['n_out']))
                final_layers.append(nn.Relu())
            
        return nn.Sequential(*final_layers)
                                    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr) 
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
        return {"optimizer": optim, "lr_scheduler": lr_sched, 'monitor': 'loss'}
    
    def forward(self, x):
        assert False
    
    def training_step(self, batch, batch_idx):
        assert False
        
    def validation_step(self, batch, batch_idx):
        assert False
        
        
    def training_step_end(self, batch_parts):
        assert False
        
    def validation_step_end(self, batch_parts):
        assert False
        
    def training_epoch_end(self, step_outputs):
        assert False
        
    def validation_epoch_end(self, step_outputs):
        assert False