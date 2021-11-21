# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:24:29 2021

@author: Arun Gautham Soundarrajan
"""

import torch
import torch.nn.functional as F
from torchmetrics import IoU
from torchsummary import summary

def meanIOU(label, pred, num_classes):
    '''
    

    Parameters
    ----------
    label : TENSOR
        The ground truth.
    pred : TENSOR
        The predicted mask.
    num_classes : INT, 
        Number of classes in the dataset. 

    Returns
    -------
    output : FLOAT
        The mean Intersection Over Union.

    '''
    
    with torch.no_grad():
        
        label = label.to('cpu')
        pred = pred.to('cpu')
        
        iou = IoU(num_classes=num_classes)
        output = iou(pred, label)
        
    return output


def pixelAcc(label, pred): 
    '''
    

    Parameters
    ----------
    label : TENSOR
        The ground truth.
    pred : TENSOR
        The predicted mask.

    Returns
    -------
    accuracy : FLOAT
        The mean Pixel Accuracy over all the classes.

    '''
    
    with torch.no_grad():
        
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        correct = torch.eq(pred, label).int()
        
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy
    
    
def count_parameters(model):
    '''
    

    Parameters
    ----------
    model : TYPE
        The pytorch model.

    Returns
    -------
    parameters : INT
        The number of trainable parameters.

    '''
    
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    
    return parameters


#summary(model, (3, 128, 128))