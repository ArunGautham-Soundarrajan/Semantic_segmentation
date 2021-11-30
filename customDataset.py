# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:49:28 2021

@author: Arun Gautham Soundarrajan
"""

import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import PIL.Image


class CustomDataset(Dataset):
    
    def __init__(self, img_dir, pixel_map, transform = None):
        '''
        

        Parameters
        ----------
        img_dir : STR
            The image directory path.
        pixel_map : BOOL
            To map the pixel only needed for FAT Dataset.
        transform : TYPE, optional
            Pytorch Transformations to apply on the image. The default is None.

        Returns
        -------
        None.

        '''
        
        self.img_dir = img_dir
        self.imgs = list(sorted(os.listdir(self.img_dir +'\\images')))
        self.masks = list(sorted(os.listdir(self.img_dir + '\\labels')))
        self.transform = transform
        self.pixel_map = pixel_map
    
    
    def __len__(self):
        '''
        

        Returns
        -------
        INT
            The length of the dataset.

        '''
        
        return len(self.imgs)
    
    def __getitem__(self, index):
        '''
        

        Parameters
        ----------
        index : INT
            Index of the image to retrieve.

        Returns
        -------
        img : TENSOR
            The image.
        mask : TENSOR
            The respective mask.

        '''
        
        image = os.path.join(self.img_dir, 'images', self.imgs[index])
        mask_path = os.path.join(self.img_dir, 'labels', self.masks[index])
        
        img = PIL.Image.open(image).convert("RGB")
        img = np.array(img)
        mask = PIL.Image.open(mask_path)
        mask = mask.convert('L')
        mask = np.array(mask)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
                    
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)
        
        if self.pixel_map == True: 
            for i in np.unique(mask):
                
                value = math.floor(i/12) * 12
                mask[mask == i] = value
             
            mask = mask/12  
            mask = mask.type(torch.LongTensor)
            
        img = T.ToTensor()(img)

        return  img, mask


# Custom Dataset for self training
class SelfTrainingDataset(Dataset):
    
    
    def __init__(self, img, mask, transform = None):
        '''
        

        Parameters
        ----------
        img : LIST (img)
            List of images.
        mask : LIST (img)
            List of pseudo labels.
        transform : TYPE, optional
            Pytorch Transformations to apply on the image. The default is None.
            
        Returns
        -------
        None.

        '''

        self.img = img
        self.mask = mask
        self.transform = transform
    
    def __len__(self):
        '''
        

        Returns
        -------
        INT
            The length of the dataset.

        '''

        return len(self.img)
    
    def __getitem__(self, index):
        '''
        

        Parameters
        ----------
        index : INT
            The index of the img and mask to retrieve.

        Returns
        -------
        img : TENSOR
            The image tensor to train.
        mask : TENSOR
            The respective mask

        '''
        
        img = self.img[index]
        mask = self.mask[index]
        
        img = np.array(img.cpu()).reshape(128,128,3)
        mask = np.array(mask.cpu()).reshape(128,128)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
       
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)
        
        return  T.ToTensor()(img), mask
