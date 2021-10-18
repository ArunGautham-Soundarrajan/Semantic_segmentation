# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:11:04 2021

@author: Arun Gautham Soundarrajan
"""


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

class CustomDataset(Dataset):
    
    def __init__(self, img_path, mask_path, transforms = None):
        
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.masks = list(sorted(os.listdir(self.mask_path)))
        self.transforms = transforms
    
    def __len__(self):
        
        return len(self.imgs)
    
    def __getitem__(self, idx):

        img = Image.open(self.img_path + self.imgs[idx]).convert("RGB")
        mask = Image.open(self.mask_path + self.masks[idx])


        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # print(num_objs)
        boxes = []
        for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])

          boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        img = self.transforms(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["labels"] = labels # Not sure if this is needed
        
     
        return img.double(), target


    
img_dir = 'PennFudanPed/PNGImages/'
mask_dir = 'PennFudanPed/PedMasks/'



transform = []
transform.append(T.ToTensor())
transform.append(T.Resize((128,128)))
transform = T.Compose(transform)

#Dataset
train_dataset = CustomDataset(img_path = img_dir, 
                          mask_path = mask_dir,
                          transforms=transform)

train_loader = DataLoader(dataset = train_dataset, 
                              batch_size= 32, 
                              shuffle = True)


dataiter = iter(train_loader)
dataiter.next()