# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:00:49 2021

@author: Arun Gautham Soundarrajan
"""

import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead
import segmentation_models_pytorch as smp


class DeepLabModel(nn.Module):
    
    def __init__(self, num_classes):
        
        super(DeepLabModel, self).__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier = DeepLabHead((2048), self.num_classes)
        
    def forward(self, x):
        
        y = self.model(x)['out']
    
        return y
    
    
class LrASPPModel(nn.Module):
    
    def __init__(self, num_classes):
        
        super(LrASPPModel, self).__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=False
                                                                              , num_classes=self.num_classes
                                                                               )
        
        #self.model.classifier = LRASPPHead(64, 64,self.num_classes, 64)
        
    def forward(self, x):
        
        y = self.model(x)['out']
    
        return y

#Unet Pretrained model    
def get_Unet(num_classes):
    model =  smp.Unet(
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=num_classes)
    return model

#PSPNet Pretrained model
def get_PSPNet(num_classes):
    model =  smp.PSPNet()(
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=num_classes)
    return model