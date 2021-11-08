# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:08:40 2021

@author: Arun Gautham Soundarrajan
"""
import torch
from models import DeepLabModel, LrASPPModel, get_Unet
import matplotlib.pyplot as plt
from evaluation_metrics import *
import os
#from main import test_dataset

def inference(model, dataset):
    
    timings = []
    iou = []
    pix_acc = []
    counter = 0
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    
    for img, mask in dataset:       
           
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

            #calculate metrics
            iou.append(meanIOU(ground_truth, pred))
            pix_acc.append(pixelAcc(ground_truth, pred))
              
            #Change the number of channels
            pred = torch.argmax(pred, dim = 1)
            
            #print(pred.shape)
            #print(img.shape)
            #print(ground_truth.shape)
            #print(img.shape)
            
            #plots
            
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
            
            #metrics = 'IoU: '+ str(iou) + '\n' + 'Pixel acc: '+ str(pixel_accuracy)
            #plt.text(0.95, 0.01, metrics)
            
            plt.savefig(os.path.join('test_plots', str(counter)))
            #print(iou)
            #print(pixel_accuracy)
            
            counter +=1
            
    return timings, iou, pix_acc


#img_path = 'Data_OCID/images/result_2018-08-20-09-43-38.png'

#inference(img_path, model_path)    
 
#plt.imshow(Image.open('Data_OCID/labels/result_2018-08-20-09-30-27.png'))        