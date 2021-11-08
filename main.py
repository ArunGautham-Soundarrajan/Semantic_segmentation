# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:49:29 2021

@author: Arun Gautham Soundarrajan
"""

import torch
import os
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from customDataset import CustomDataset
from models import DeepLabModel, LrASPPModel, get_Unet
from tqdm import tqdm
#import matplotlib.pyplot as plt
from evaluation_metrics import meanIOU, pixelAcc, count_parameters
from plots import *
from inference import inference
#import timeit

#create necessary directories
#get current working directory
cwd = os.getcwd()

#create a directory if doesnt exist
if not os.path.exists('test_plots'):
    os.makedirs('test_plots')
    
#create a directory if doesnt exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS):
    
    #list to store loss
    val_loss = []
    total_loss = []
    iou_train = []
    meanioutrain = []
    mean_pixacc_train = []
    pixelacctrain = []
    iou_test = []
    meanioutest = []
    mean_pixacc_test = []
    pixelacctest = []
    #mean_train_time = []
        
    for epoch in range(EPOCHS):
        
        #set the model to train mode
        model.train()
        
        loop = tqdm(train_loader, desc= "Training Epoch")
        val_loop = tqdm(test_loader, desc = "Validation Epoch")
        
        #training set
        for b, (img, mask) in enumerate(loop):
            
            #torch.cuda.empty_cache()
            
            #change to device
            img = img.to(device = DEVICE)
            mask = mask.to(device = DEVICE)
            
            #start the timer
            #start = timeit.default_timer()
            
            #forward pass
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #stop the timer
            #stop = timeit.default_timer()
            #train_time = stop - start
            
            #evalutaion metrics
            iou_train.append(meanIOU(mask, y_pred))
            pixelacctrain.append(pixelAcc(mask, y_pred))
            #mean_train_time.append(train_time) 
                       
            #display the loss
            loop.set_postfix({'Epoch': epoch+1  ,
                              'Loss': loss.item(),
                              'Mean IOU': iou_train.item(),
                              'Pixel Acc': pixelacctrain.item(),
                              #'Train time image': np.mean(mean_train_time)
                              })
         
            
        #append the loss    
        total_loss.append(loss.item())
        meanioutrain.append(iou_train.item())
        mean_pixacc_train.append(pixelacctrain.item())
         
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
                iou_test.append(meanIOU(mask, val_pred))
                pixelacctest.append(pixelAcc(mask, val_pred))
            
                #display in tqdm
                val_loop.set_postfix({'Epoch ': epoch+1  ,
                              'Loss ': v_loss.item(),
                              'Mean IOU ': np.mean(iou_test),
                              'Pixel Acc :': np.mean(pixelacctest)
                              })
                
            #append the loss    
            val_loss.append(v_loss.item())
            meanioutest.append(np.mean(iou_test))
            mean_pixacc_test.append(np.mean(pixelacctest))
            
    torch.save(model.state_dict(),'deeplab_model.pth')
    
    metrics = (total_loss, val_loss,  meanioutrain, mean_pixacc_train, meanioutest, mean_pixacc_test)
    
    return model, metrics
            
if __name__ == "__main__":

    #Data directory
    img_dir = 'Data_OCID'
    
    #Trainging Params
    BATCH_SIZE = 16
    EPOCHS = 50
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
    dataset = CustomDataset(img_dir = img_dir,
                            transform=transform)
    
    #train test split
    train_dataset , test_dataset = random_split(dataset,(900,166))
    
        
    #DataLoader
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size= BATCH_SIZE, 
                              shuffle = True)
    
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size= BATCH_SIZE)
    
    #Model
    #model = DeepLabModel(num_classes=23).to(device=DEVICE)
    #model = LrASPPModel(num_classes=23).to(device=DEVICE)
    model = get_Unet(num_classes=23).to(device=DEVICE)
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    
    
    #number of parameters
    print('Number of trainable parameters :', count_parameters(model))
    
    #train
    trained_model, metrics = train(model, train_loader, test_loader,
                                                criterion, optimizer, EPOCHS)
    
    total_loss, val_loss,  meanioutrain, mean_pixacc_train, meanioutest, mean_pixacc_test = metrics
    
    #loss plot
    loss_plot(total_loss, val_loss)
    
    #mean iou plot
    mean_iou_plot( EPOCHS, meanioutrain, meanioutest, 'Mean IoU')
    
    #mean pixel accuracy
    pixel_acc_plot( EPOCHS, mean_pixacc_train, mean_pixacc_test, 'Mean Pixel Accuracy')
    
    counter = 0       
    for img, mask in test_dataset:        
        inference(img, mask, trained_model, str(counter))
        counter += 1 