# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:49:29 2021

@author: Arun Gautham Soundarrajan
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from customDataset import CustomDataset
from deepLabModel import DeepLabModel, LrASPPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation_metrics import meanIOU, pixelAcc, count_parameters
import timeit


def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS):
    
    #list to store loss
    val_loss = []
    total_loss = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    mean_train_time = []
        
    for epoch in range(EPOCHS):
        
        #set the model to train mode
        model.train()
        
        loop = tqdm(train_loader, desc= "Training Epoch")
        val_loop = tqdm(test_loader, desc = "Validation Epoch")
        

        #training set
        for b, (img, mask) in enumerate(loop):
            
            torch.cuda.empty_cache()
            
            #change to device
            img = img.to(device = DEVICE)
            mask = mask.to(device = DEVICE)
            
            #start the timer
            start = timeit.default_timer()
            
            #forward pass
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #stop the timer
            stop = timeit.default_timer()
            train_time = stop - start
            
            #evalutaion metrics
            meanioutrain.append(meanIOU(mask, y_pred))
            pixelacctrain.append(pixelAcc(mask, y_pred))
            mean_train_time.append(train_time) 
                       
            #display the loss
            loop.set_postfix({'Epoch': epoch+1  ,
                              'Loss': loss.item(),
                              'Mean IOU': np.mean(meanioutrain),
                              'Pixel Acc': np.mean(pixelacctrain),
                              'Train time image': np.mean(mean_train_time)
                              })
              
        #append the loss    
        total_loss.append(loss.item())
        
        #validation set
        with torch.no_grad():
            
            #set the model to eval mode
            model.eval()
            for j, (img, mask) in enumerate(val_loop):
                
                #Set to device
                img = img.to(device = DEVICE)
                mask = mask.to(device = DEVICE)
                
                #validation prediction and loss
                val_pred = model(img)
                v_loss = criterion(val_pred, mask)
                
                #accuracy and iou
                meanioutest.append(meanIOU(mask, val_pred))
                pixelacctest.append(pixelAcc(mask, val_pred))
            
                #display in tqdm
                val_loop.set_postfix({'Epoch ': epoch+1  ,
                              'Loss ': v_loss.item(),
                              'Mean IOU ': np.mean(meanioutest),
                              'Pixel Acc :': np.mean(pixelacctest)
                              })
                
            #append the loss    
            val_loss.append(v_loss.item())
            
    torch.save(model.state_dict(),'deeplab_model.pth')
    
    return model, total_loss, val_loss
            
if __name__ == "__main__":

    #Data directory
    img_dir = 'data'
    #mask_dir = 'archive/TrayDataset/yTrain/'
    val_img_dir = 'archive/TrayDataset/XTest/'
    #val_mask_dir = 'archive/TrayDataset/yTest/'
    
    #Trainging Params
    BATCH_SIZE = 16
    EPOCHS = 5
    IMG_SIZE = 128
    SEED = 24
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    #SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    #Preprocessing
    transform = T.Compose([
            T.Resize((IMG_SIZE,IMG_SIZE)),
            ])
    
    #Dataset
    train_dataset = CustomDataset(img_path = img_dir,
                                  transform=transform)
    
    val_dataset = CustomDataset(img_path = val_img_dir,
                                transform=transform)
    
    #DataLoader
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size= BATCH_SIZE, 
                              shuffle = True)
    
    test_loader = DataLoader(dataset = val_dataset,
                             batch_size= BATCH_SIZE)
    
    #Model
    #model = DeepLabModel(num_classes=43).to(device=DEVICE)
    model = LrASPPModel(num_classes=43).to(device=DEVICE)
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    
    
    #number of parameters
    print('Number of trainable parameters :',count_parameters(model))
    
    #train
    trained_model, train_loss, val_loss = train(model, train_loader, test_loader,
                                                criterion, optimizer, EPOCHS)
    
    #plot the loss
    plt.figure(figsize=(8,8))
    plt.title('Loss per Epoch', fontsize=22)
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(frameon=False)
    plt.savefig('loss plot')