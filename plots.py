# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:42:08 2021

@author: Arun Gautham Soundarrajan
"""

import os
import matplotlib.pyplot as plt


#get current working directory
cwd = os.getcwd()

#create a directory if doesnt exist
if not os.path.exists('plots'):
    os.makedirs('plots')

#function to create a loss plot
def loss_plot(train_loss,val_loss, desc):
    '''
    

    Parameters
    ----------
    train_loss : LIST(Float)
        The training loss for each epoch.
    val_loss : LIST(Float)
        The validation loss for each epoch.
    desc : STR
        The title for the plot.

    Returns
    -------
    None.

    '''

    plt.figure(figsize=(8,8))
    plt.title('Loss per Epoch', fontsize=22)
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots', str(desc).replace(' ', '_')))

    
#function to plot the Mean Iou
def mean_iou_plot(epochs, mean_iou_train, mean_iou_test, desc):
    '''
    

    Parameters
    ----------
    epochs : INT
        The number of epochs.
    mean_iou_train : LIST(float)
        The mean Intersection over Union for each epoch for training.
    mean_iou_test : LIST(float)
        The mean Intersection over Union for each epoch for testing.
    desc : STR
        The title for the plot.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(8,8))
    plt.title(desc, fontsize=22)
    plt.plot(mean_iou_train, label='train')
    plt.plot(mean_iou_test, label='test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Mean Intersection Over Union', fontsize=16)
    #plt.xticks(list(range(1,epochs+1)))
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots', str(desc).replace(' ', '_')))
    
#function to plot pixel accuracy     
def pixel_acc_plot(epochs, mean_pixelacc_train, mean_pixelacc_test, desc):
    '''
    

    Parameters
    ----------
    epochs : INT
        The number of epochs.
    mean_pixelacc_train : LIST(float)
        The mean Pixel accuracy for each epochs for training.
    mean_pixelacc_test : LIST(float)
        The mean Pixel Accuracy for each epochs for testing.
    desc : TYPE
        The title for the plot.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(8,8))
    plt.title(desc, fontsize=22)
    plt.plot(mean_pixelacc_train, label='train')
    plt.plot(mean_pixelacc_test, label='test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Mean Pixel Accuracy', fontsize=16)
    
    plt.legend(frameon=False)
    plt.savefig(os.path.join(cwd, 'plots', str(desc).replace(' ', '_')))