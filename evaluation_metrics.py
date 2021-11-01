# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:24:29 2021

@author: Arun Gautham Soundarrajan
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics import IoU


# =============================================================================
# def meanIOU(label, pred, num_classes=43):
#     
#     #no grad
#     with torch.no_grad():
#         
#         #softmax and squeeze the dimensions
#         pred = F.softmax(pred, dim=1)              
#         pred = torch.argmax(pred, dim=1).squeeze(1)
#         
#         #placeholders
#         iou_list = list()
#         present_iou_list = list()
#     
#         pred = pred.view(-1)
#         label = label.view(-1)
#         # Note: Following for loop goes from 0 to (num_classes-1)
#         # and ignore_index is num_classes, thus ignore_index is
#         # not considered in computation of IoU.
#         
#         for sem_class in range(num_classes):
#             pred_inds = (pred == sem_class)
#             target_inds = (label == sem_class)
#             if target_inds.long().sum().item() == 0:
#                 iou_now = float('nan')
#             else: 
#                 intersection_now = (pred_inds[target_inds]).long().sum().item()
#                 union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
#                 iou_now = float(intersection_now) / float(union_now)
#                 present_iou_list.append(iou_now)
#             iou_list.append(iou_now)
#             
#     return np.mean(present_iou_list)
# 
# =============================================================================


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
