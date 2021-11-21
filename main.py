# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:49:29 2021

@author: Arun Gautham Soundarrajan
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import argparse
import math

from customDataset import CustomDataset
from models import get_Unet, get_PSPNet, get_DeepLabv3_plus
from evaluation_metrics import meanIOU, pixelAcc, count_parameters
from plots import *
from inference import inference


def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, num_classes, model_name):
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
            
if __name__ == "__main__":
        
    #Creating the parser
    my_parser = argparse.ArgumentParser(description= 'Specify the Model to train')
    
    #add the arguments
    my_parser.add_argument('Model', 
                           metavar='model',
                           type = str,
                           help = 'b: Baseline \t d: Deep Lab \t p: PSPNet'
                           )
    
    my_parser.add_argument('Data', 
                           metavar='data',
                           type = int,
                           help = '1: ARID20 \t 2: FAT data'
                           )
    
    args= my_parser.parse_args()
    
    #store the arguments in the variable
    model_to_train = args.Model
    data_to_use = args.Data
        
    #create necessary directories
    #get current working directory
    cwd = os.getcwd()
    
    #create a directory if doesnt exist
    if not os.path.exists('test_plots'):
        os.makedirs('test_plots')
        
    #create a directory if doesnt exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    #create a directory if doesnt exist
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
        
    if not os.path.exists('models'):
        os.makedirs('models')
        
    
    #Trainging Params
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = 128
    SEED = 29
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    print(f'Training using {DEVICE}') 
    if DEVICE == 'cuda':
        DEVICE_NUM = torch.cuda.current_device()
        print('Device name : ',torch.cuda.get_device_name(DEVICE_NUM))    
    
    #SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    #Preprocessing
    transform = T.Compose([
            T.Resize((IMG_SIZE,IMG_SIZE)),
            ])
    
    #Number of classes
    if data_to_use == 1:
        NUM_CLASSES = 23
        
    elif data_to_use == 2:
        NUM_CLASSES = 22
              
    #Select the model to train    
    if model_to_train == 'b':
        
        model = get_Unet(num_classes=NUM_CLASSES).to(device=DEVICE)
        model_name = 'Unet'
        
    elif model_to_train == 'd':
        
        model = get_DeepLabv3_plus(num_classes=NUM_CLASSES).to(device=DEVICE)
        model_name = 'Deep_lab_v3+'
    
    elif model_to_train == 'p':
        
        model = get_PSPNet(num_classes=NUM_CLASSES).to(device=DEVICE)
        model_name = 'PSPNet'
    
    #Data directory
    if data_to_use == 1:
        img_dir = 'Data_OCID'
        model_name = model_name + '_ARID20'
        dataset = CustomDataset(img_dir = img_dir,
                                pixel_map=False,
                                transform=transform)
        
    elif data_to_use == 2:
        img_dir = 'fat_data'
        model_name = model_name + '_FAT'
        dataset = CustomDataset(img_dir = img_dir,
                                pixel_map=True,
                                transform=transform)
                
    train_percent = math.floor(dataset.__len__()*0.9)
    test_percent = dataset.__len__() - train_percent
    
    #train test split 90:10
    train_dataset , test_dataset = random_split(dataset,( train_percent, test_percent))
     
    #DataLoader
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size= BATCH_SIZE, 
                              shuffle = True)
    
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size= BATCH_SIZE)
        
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    
    #number of parameters
    print('\nNumber of trainable parameters :', count_parameters(model))
    
    #train
    trained_model, metrics = train(model, train_loader, test_loader,
                                   criterion, optimizer, EPOCHS, NUM_CLASSES, model_name)
    
    total_loss, val_loss,  meanioutrain, mean_pixacc_train, meanioutest, mean_pixacc_test = metrics
    
    #store metrics into a csv file
    metrics_df = pd.DataFrame(total_loss, columns = ['Train_loss'])
    metrics_df['val_loss'] = val_loss
    metrics_df['Iou_train'] = meanioutrain
    metrics_df['Iou_test'] = meanioutest
    metrics_df['Pixel_acc_train'] = mean_pixacc_train
    metrics_df['Pixel_acc_test'] = mean_pixacc_test
    metrics_df.to_csv( os.path.join('metrics', model_name + '_.csv'), index = False)
    
    #loss plot
    loss_plot(total_loss, val_loss, model_name + '_Loss')
    
    #mean iou plot
    mean_iou_plot(EPOCHS, meanioutrain, meanioutest, model_name +'_Mean IoU')
    
    #mean pixel accuracy
    pixel_acc_plot(EPOCHS, mean_pixacc_train, mean_pixacc_test, model_name +'_Mean Pixel Accuracy')
    
    #inference    
    #inference_time, iou, pix_acc = inference(trained_model, test_dataset)
    