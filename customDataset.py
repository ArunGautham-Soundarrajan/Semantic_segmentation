# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:49:28 2021

@author: Arun Gautham Soundarrajan
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class CustomDataset(Dataset):
    
    def __init__(self, img_path, mask_path, transform = None):
        
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.masks = list(sorted(os.listdir(self.mask_path)))
        self.transform = transform
    
    def __len__(self):
        
        return len(self.imgs)
    
    def __getitem__(self, index):
        
        img = Image.open(self.img_path + self.imgs[index]).convert("RGB")
        mask = Image.open(self.mask_path + self.masks[index])
        mask = mask.convert('L')
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
                    
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)
        
        img = T.ToTensor()(img)
               
        return  img, mask


