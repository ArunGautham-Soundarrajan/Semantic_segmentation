# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:14:53 2021

@author: Arun Gautham Soundarrajan
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as T


from models import get_Unet, get_PSPNet, get_DeepLabv3_plus
from evaluation_metrics import meanIOU, pixelAcc, count_parameters


def prediction(model, test_dataset, criterion, NUM_CLASSES):
    
    img_list = []
    mask_list = []
    
    m_iou = []
    pixel_acc = []
    
    with torch.no_grad():

        #set the model to eval mode
        model.eval()
        
        for img, mask in test_dataset:
            
            if torch.cuda.is_available():
                DEVICE = 'cuda'
            else:
                DEVICE = 'cpu'
        
            #Set to device
            img = img.to(device = DEVICE)
            mask = mask.to(device = DEVICE)
        
            #pseudo label
            val_pred = model(img.unsqueeze(0))
            
            m_iou.append(meanIOU(mask, val_pred, NUM_CLASSES))
            pixel_acc.append(pixelAcc(mask, val_pred))
            
            #store img and pseudo labels
            img_list.append(img)
            mask_list.append(torch.argmax(val_pred, dim = 1))
    
    print('Number of test images :', len(img_list))        
    print('\nMean IoU :', np.mean(m_iou))
    print('Mean Pixel Accuracy :', np.mean(pixel_acc))

    return img_list, mask_list

