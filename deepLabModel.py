# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:00:49 2021

@author: Arun Gautham Soundarrajan
"""

import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead

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
                                                                              , num_classes=43
                                                                               )
        
        #self.model.classifier = LRASPPHead(64, 64,self.num_classes, 64)
        
    def forward(self, x):
        
        y = self.model(x)['out']
    
        return y