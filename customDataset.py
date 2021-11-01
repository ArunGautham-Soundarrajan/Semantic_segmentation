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
    
    def __init__(self, img_dir, transform = None):
        
        self.img_dir = img_dir
        self.imgs = list(sorted(os.listdir(self.img_dir +'\\images')))
        self.masks = list(sorted(os.listdir(self.img_dir + '\\labels')))
        self.transform = transform
    
    def __len__(self):
        
        return len(self.imgs)
    
    def __getitem__(self, index):
        
        image = os.path.join(self.img_dir, 'images', self.imgs[index])
        mask_path = os.path.join(self.img_dir, 'labels', self.masks[index])
        
        img = Image.open(image).convert("RGB")
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
                    
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)
        
        img = T.ToTensor()(img)
               
        return  img, mask

transform = T.Compose([
            T.Resize((128, 128)),
            ])
dataset = CustomDataset(img_dir = 'Data_OCID',
                        transform=transform)

#the dimensions of the image are (480,640, 3)
#the dimensions of the mask are (480,640)