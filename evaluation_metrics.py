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

#Reference
'''

This part of the code for calculating Mean Pixel Accuracy is taken from,
#https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py

'''   

EPS = 1e-10


def nanmean(x):
    
    """
    
    
    Computes the arithmetic mean ignoring any NaNs.
       
    """
    
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
                        num_classes * true[mask] + pred[mask],
                        minlength=num_classes ** 2,
                        ).reshape(num_classes, num_classes).float()
    
    return hist

def pixelAcc(label, pred, num_classes):
    """
    
    
    
    Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    
    Args:
        hist: confusion matrix.
        
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
        
    """
    
    
    label = label
    pred = torch.argmax(pred, dim = 1)
    
    hist = _fast_hist(label, pred, num_classes)
    
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    
    return avg_per_class_acc.cpu()

#######
   
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