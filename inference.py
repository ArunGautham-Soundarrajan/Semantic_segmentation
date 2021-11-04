# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:08:40 2021

@author: Arun Gautham Soundarrajan
"""
import torch
import torchvision.transforms as T
from PIL import Image
from models import DeepLabModel, LrASPPModel, get_Unet
import matplotlib.pyplot as plt

def inference(img_path, model_path):
    
    #load the image and transform it
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
            T.Resize((128,128)),
            T.ToTensor()
            ])
    
    img = transform(img)
    
    #load the trained model
    model = get_Unet(23)
    model.load_state_dict(torch.load(model_path))
    
    with torch.no_grad():
        model.eval()
        
        #Set device to Cuda if available
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
            
        model = model.to(DEVICE)
        img = img.to(DEVICE)
        
        
        #Pass the image
        pred = model(img.unsqueeze(0))
        
        #convert it back to cpu
        pred = pred.cpu()
        
        #Change the number of channels
        pred = torch.argmax(pred, dim = 1)
        
        #display the image
        plt.imshow(pred.view(128,128,1))
        print(pred.shape)
        
   
        
img_path = 'Data_OCID/images/result_2018-08-20-09-30-36.png'
model_path = 'deeplab_model.pth'
inference(img_path, model_path)    
 
#plt.imshow(Image.open('Data_OCID/labels/result_2018-08-20-09-30-27.png')   )        