# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:31:55 2021

@author: Arun Gautham Soundarrajan
"""

import os
import numpy as np
import torch

from tqdm import tqdm
from evaluation_metrics import meanIOU, pixelAcc

def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, num_classes, DEVICE, model_name):
    '''
    
    Parameters
    ----------
    model : TYPE
        Pytorch model to train.
    train_loader : TYPE
        Data loader for training set.
    test_loader : TYPE
        Data loader for validation/testing set.
    criterion : TYPE
        Loss Function.
    optimizer : TYPE
        Optimizer of choice.
    EPOCHS : INT
        The number of Epochs to train for.
    num_classes : INT
        The number of classes.
    DEVICE : STR
        Cuda or cpu
    model_name : STR
        The name of model for saving it.

    Returns
    -------
    model : TYPE
        Trained model.
    metrics : TUPLE
        Tuple containing all the evaluation metrics.

    '''
    
    
    #list to store loss
    val_loss = []
    total_loss = []
    
    meanioutrain = []
    mean_pixacc_train = []
    
    meanioutest = []
    mean_pixacc_test = []
            
    for epoch in range(EPOCHS):
        
        iou_train = []
        pixelacctrain = []
        iou_test = []
        pixelacctest = []
                
        #set the model to train mode
        model.train()
        
        loop = tqdm(train_loader, desc= "Training Epoch")
        val_loop = tqdm(test_loader, desc = "Validation Epoch")
        
        #training set
        for b, (img, mask) in enumerate(loop):
                        
            #change to device
            img = img.to(device = DEVICE)
            mask = mask.to(device = DEVICE)
            
            #forward pass
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
            #evalutaion metrics
            iou_train.append(meanIOU(mask, y_pred, num_classes))
            pixelacctrain.append(pixelAcc(mask, y_pred))
            #mean_train_time.append(train_time) 
                       
            #display the loss
            loop.set_postfix({'Epoch': epoch+1  ,
                              'Loss': loss.item(),
                              'Mean IOU': np.mean(iou_train),
                              'Pixel Acc': np.mean(pixelacctrain),
                              })
         
        #append the loss    
        total_loss.append(loss.item())
        meanioutrain.append(np.mean(iou_train))
        mean_pixacc_train.append(np.mean(pixelacctrain))
         
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
                iou_test.append(meanIOU(mask, val_pred, num_classes))
                pixelacctest.append(pixelAcc(mask, val_pred))
            
                #display in tqdm
                val_loop.set_postfix({'Epoch ': epoch+1,
                              'Loss ': v_loss.item(),
                              'Mean IOU ': np.mean(iou_test),
                              'Pixel Acc :': np.mean(pixelacctest)
                              })
                
            #append the loss    
            val_loss.append(v_loss.item())
            meanioutest.append(np.mean(iou_test))
            mean_pixacc_test.append(np.mean(pixelacctest))
            
    torch.save(model.state_dict(), os.path.join('models' , (model_name +'.pth')))
    
    metrics = (total_loss, val_loss,  meanioutrain, mean_pixacc_train, meanioutest, mean_pixacc_test)
    
    return model, metrics


def self_trainer(model, test_loader, criterion, optimizer, EPOCHS, num_classes, DEVICE, model_name):
    '''
    
    Parameters
    ----------
    model : TYPE
        Pytorch model to train.
    test_loader : TYPE
        Data loader for validation/testing set.
    criterion : TYPE
        Loss Function.
    optimizer : TYPE
        Optimizer of choice.
    EPOCHS : INT
        The number of Epochs to train for.
    num_classes : INT
        The number of classes.
    DEVICE : STR
        Cuda or cpu
    model_name : STR
        The name of model for saving it.

    Returns
    -------
    model : TYPE
        Trained model.

    '''
    
    #list to store loss
    total_loss = []
    
    meanioutrain = []
    mean_pixacc_train = []
    
    for epoch in range(EPOCHS):
        
        iou_train = []
        pixelacctrain = []

        #set the model to train mode
        model.train()
        
        loop = tqdm(test_loader, desc = "Self Training Epoch")
        
        #training set
        for b, (img, mask) in enumerate(loop):
            
            #change to device
            img = img.to(device = DEVICE)
            mask = mask.to(device = DEVICE)
            
            #forward pass
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
            #evalutaion metrics
            iou_train.append(meanIOU(mask, y_pred, num_classes))
            pixelacctrain.append(pixelAcc(mask, y_pred))
            #mean_train_time.append(train_time) 
                       
            #display the loss
            loop.set_postfix({'Epoch': epoch+1  ,
                              'Loss': loss.item(),
                              'Mean IOU': np.mean(iou_train),
                              'Pixel Acc': np.mean(pixelacctrain),
                              })
         
        #append the loss    
        total_loss.append(loss.item())
        meanioutrain.append(np.mean(iou_train))
        mean_pixacc_train.append(np.mean(pixelacctrain))

    torch.save(model.state_dict(), os.path.join('models' , ('st_' + model_name +'.pth')))
    
    return model
    
    
    
    
    
    