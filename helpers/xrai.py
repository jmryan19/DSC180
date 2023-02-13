import saliency.core as saliency
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn

class XRai(nn.Module):
    def __init__(self, ResNet):
        super(XRai, self).__init__()
    
        self.net = ResNet
    
    def call_model_function(images, call_model_args=None, expected_keys=None):
        images = torch.movedim(torch.tensor(images), 3, 1).requires_grad_(True)
        outputs = net(images.to('cuda')).to('cpu')

        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

    def forward(self, im):
        def call_model_function(images, call_model_args=None, expected_keys=None):
            images = torch.movedim(torch.tensor(images), 3, 1).requires_grad_(True)
            outputs = self.net(images.to('cuda')).to('cpu')
            if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
                grads = torch.movedim(grads[0], 1, 3)
                gradients = grads.detach().numpy()
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

        print(im)
        print(im.shape)
        im = torch.movedim(im, 0, 2).numpy()
        xrai_object = saliency.XRAI()

        xrai_attributions = xrai_object.GetMask(im, call_model_function, batch_size=10)
        
        return xrai_attributions