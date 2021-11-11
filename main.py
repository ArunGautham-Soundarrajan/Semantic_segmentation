# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:49:29 2021

@author: Arun Gautham Soundarrajan
"""

import torch
import os
import pandas as pd
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
import argparse
#import timeit

def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, model_name):
    
    #list to store loss
    val_loss = []
    total_loss = []
    
    meanioutrain = []
    mean_pixacc_train = []
    
    
    meanioutest = []
    mean_pixacc_test = []
    
    #mean_train_time = []
        
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
                              'Mean IOU': np.mean(iou_train),
                              'Pixel Acc': np.mean(pixelacctrain),
                              #'Train time image': np.mean(mean_train_time)
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
                           help = 'b: Baseline \t d: Deep Lab')
    
    args= my_parser.parse_args()
    
    model_to_train = args.Model
    
    
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
        
    #print(model_to_train)

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
    
    if model_to_train == 'b':
        
        model = get_Unet(num_classes=23).to(device=DEVICE)
        model_name = 'Unet'
        
    elif model_to_train == 'd':
        
        model = DeepLabModel(num_classes=23).to(device=DEVICE)
        model_name = 'Deep_lab'
    
    #Model
    #model = DeepLabModel(num_classes=23).to(device=DEVICE)
    #model = LrASPPModel(num_classes=23).to(device=DEVICE)
    #model = get_Unet(num_classes=23).to(device=DEVICE)
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    
    #number of parameters
    print('Number of trainable parameters :', count_parameters(model))
    
    #train
    trained_model, metrics = train(model, train_loader, test_loader,
                                                criterion, optimizer, EPOCHS, model_name)
    
    total_loss, val_loss,  meanioutrain, mean_pixacc_train, meanioutest, mean_pixacc_test = metrics
    
    #store metrics into a csv file
    metrics_df = pd.DataFrame(total_loss, columns = ['Train_loss'])
    metrics_df['val_loss'] = val_loss
    metrics_df['Iou_train'] = meanioutrain
    metrics_df['Iou_test'] = meanioutest
    metrics_df['Pixel_acc_train'] = mean_pixacc_train
    metrics_df['Pixel_acc_test'] = mean_pixacc_test
    metrics_df.to_csv( os.path.join('metrics', 'baseline.csv'), index = False)
    
    #loss plot
    loss_plot(total_loss, val_loss)
    
    #mean iou plot
    mean_iou_plot(EPOCHS, meanioutrain, meanioutest, model_name +'_Mean IoU')
    
    #mean pixel accuracy
    pixel_acc_plot(EPOCHS, mean_pixacc_train, mean_pixacc_test, model_name +'_Mean Pixel Accuracy')
    
    #inference    
    inference_time, iou, pix_acc = inference(trained_model, test_dataset)
    