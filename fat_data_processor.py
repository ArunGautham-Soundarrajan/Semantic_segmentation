#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import random 

base_path = os.getcwd()
data_path = os.path.join(os.getcwd(), 'fat', 'fat','mixed','kitchen_0')  

if not os.path.exists(os.path.join(base_path, 'fat_data')):
    os.mkdir(os.path.join(base_path, 'fat_data'))

dest_path = os.path.join(base_path, 'fat_data')

if not os.path.exists(os.path.join(dest_path, 'images')):    
    os.mkdir(os.path.join(dest_path, 'images'))
if not os.path.exists(os.path.join(dest_path, 'labels')):       
    os.mkdir(os.path.join(dest_path, 'labels'))

for img in os.listdir(data_path):
    
    if img.endswith('.jpg'):       
        shutil.copyfile(os.path.join(data_path, img),
                        os.path.join(dest_path, 'images',img))
        
    elif img.endswith('.png') and 'seg' in img:    
        shutil.copyfile(os.path.join(data_path, img),
                        os.path.join(dest_path, 'labels',img))
