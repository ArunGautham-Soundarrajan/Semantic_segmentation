# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:00:49 2021

@author: Arun Gautham Soundarrajan
"""

import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp

#Unet Pretrained model    
def get_Unet(num_classes):
    '''
    

    Parameters
    ----------
    num_classes : INT
        The number of classes in the dataset.

    Returns
    -------
    model : TYPE
        UNet model with pretrained weights on imagenet.

    '''
     
    model =  smp.Unet(
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=num_classes)
    return model

#PSPNet Pretrained model
def get_PSPNet(num_classes):
    '''
    

    Parameters
    ----------
    num_classes : INT
        The number of classes in the dataset.

    Returns
    -------
    model : TYPE
        PSP Net model with pretrained weights on imagenet.

    '''
    
    model =  smp.PSPNet(
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=num_classes)
    return model

def get_DeepLabv3_plus(num_classes):
    '''
    

    Parameters
    ----------
    num_classes : INT
        The number of classes in the dataset.

    Returns
    -------
    model : TYPE
        Deep Lab V3 Plus model with pretrained weights on imagenet.

    '''
    
    model =  smp.DeepLabV3Plus(
                 encoder_name='tu-xception41',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=num_classes)
    return model