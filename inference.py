# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:08:40 2021

@author: Arun Gautham Soundarrajan
"""

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from evaluation_metrics import *
from models import *

def inference(model, dataset, num_classes, store = False):
    '''
    

    Parameters
    ----------
    model : TYPE
        The trained model.
    dataset : TYPE
        The test dataset to test the model.
    store : BOOL, optional
        Set true if wanted to save the plots. The default is False.

    Returns
    -------
    timings : LIST(Float)
        The inference time for each image.
    iou : LIST(Float)
        MeanIoU for each image.
    pix_acc : LIST(Float)
        Mean Pixel Accuracy for each image.

    '''
    
    timings = []
    counter = 0
    
    #Inference timer
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    #Set the TQDM
    loop = tqdm(dataset, desc= "Inference")
    
    for img, mask in loop:       
           
        with torch.no_grad():
            
            model.eval()
            
            #Set device to Cuda if available
            if torch.cuda.is_available():
                DEVICE = 'cuda'
            else:
                DEVICE = 'cpu'
                 
            #convert to device    
            model = model.to(DEVICE)
            image = img.to(DEVICE)
            ground_truth = mask.to(DEVICE)
            
            #start the timer
            starter.record()
            
            #Pass the image
            pred = model(image.unsqueeze(0))
            
            #end the timer
            ender.record()
            
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
              
            #Change the number of channels
            pred = torch.argmax(pred, dim = 1)
                        
            #plots
            if(store == True):
                
                plt.subplot(1,3,1)
                plt.gca().set_title('Original Image')
                plt.imshow(img.cpu().permute(1,2,0))
                plt.axis('off')
                
                #display the image
                plt.subplot(1,3,2)
                plt.gca().set_title('Predicted')
                plt.imshow(pred.cpu().permute(1,2,0))
                plt.axis('off')
                
                plt.subplot(1,3,3)
                plt.gca().set_title('Ground Truth')
                plt.imshow(ground_truth.cpu())
                plt.axis('off')
                plt.savefig(os.path.join('test_plots', str(counter)))
            
            counter +=1
            
        loop.set_postfix({
                          'Time in ms': str(timings[-1]),
                          })
            
    return timings

