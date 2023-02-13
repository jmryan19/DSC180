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
from helpers.supernet import *

class NetworkGradCAM(nn.Module):
    def __init__(self, ResNet):
        super(NetworkGradCAM, self).__init__()
        self.encoder_pre_list = ResNet.layers[:-1]
        self.encoder_post_list = ResNet.layers[-1]
        #self.encoder_list = ResNet.feature_extractor_list
        # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.regresser = ResNet.regresser
        
        self.gradients = 0
        
    # hook for the gradients of the activations
    def gradients_hook(self, grad):
        self.gradients = grad
        
    def zero_grads(self):
        self.gradients = 0
    
    def forward(self, x):
        # encoding each feature
        z = self.encode(x)
        # concatenating the encodings
        #h = z.register_hook(self.activations_hook)
        
        # classification on the encodings
        y_hat = self.regresser(z)
        
        return y_hat
    
    def encode(self, x):  
        # encoding each feature in the data
        #print(x_.shape)
        #print(x_.unsqueeze(0).shape)
        #print(x_.flatten(0).shape)
        z = self.encoder_pre_list(x)
        h = z.register_hook(self.gradients_hook)
        z = self.encoder_post_list(z).flatten(1)
            
        # to check classifier input size
        # print([z_.shape for z_ in z])
        return z
    
    # method for gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        # encoding each feature in the data 
        z = self.encoder_pre_list(x)

        return z