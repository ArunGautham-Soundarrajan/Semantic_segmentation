# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:42:08 2021

@author: Arun Gautham Soundarrajan
"""

import matplotlib.pyplot as plt
#import seaborn as sns
import os

#get current working directory
cwd = os.getcwd()

#create a directory if doesnt exist
if not os.path.exists('plots'):
    os.makedirs('plots')

#function to create a loss plot
def loss_plot(train_loss,val_loss):

    plt.figure(figsize=(8,8))
    plt.title('Loss per Epoch', fontsize=22)
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots','loss plot'))

    
#function to plot the Mean Iou
def mean_iou_plot(epochs, mean_iou_train, mean_iou_test, desc):
    
    plt.figure(figsize=(8,8))
    plt.title(desc, fontsize=22)
    plt.plot(mean_iou_train, label='Mean IoU train')
    plt.plot(mean_iou_test, label='Mean IoU test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Mean Intersection Over Union', fontsize=16)
    #plt.xticks(list(range(1,epochs+1)))
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots', str(desc).replace(' ', '_')))
    
#function to plot pixel accuracy     
def pixel_acc_plot(epochs, mean_pixelacc_train, mean_pixelacc_test, desc):
    
    plt.figure(figsize=(8,8))
    plt.title(desc, fontsize=22)
    plt.plot(mean_pixelacc_train, label='Pixel acc train')
    plt.plot(mean_pixelacc_test, label='Pixel acc test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Mean Pixel Accuracy', fontsize=16)
    
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots', str(desc).replace(' ', '_')))