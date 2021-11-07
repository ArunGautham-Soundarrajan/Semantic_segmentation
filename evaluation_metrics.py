# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:24:29 2021

@author: Arun Gautham Soundarrajan
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics import IoU


def meanIOU(label, pred, num_classes = 23):
    
    with torch.no_grad():
        
        label = label.to('cpu')
        pred = pred.to('cpu')
        
        iou = IoU(num_classes=num_classes)
        output = iou(pred, label)
        
        
    return output


def pixelAcc(label, pred):  
    
    with torch.no_grad():
        
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        correct = torch.eq(pred, label).int()
        
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
