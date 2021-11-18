#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import random 

base_path = os.getcwd()
data_path = os.path.join(os.getcwd(), 'OCID-dataset', 'OCID-dataset','ARID20')  
os.mkdir(os.path.join(base_path, 'Data_OCID'))

dest_path = os.path.join(os.path.join(base_path, 'Data_OCID'))

os.mkdir(os.path.join(dest_path, 'images'))
os.mkdir(os.path.join(dest_path, 'labels'))

background = ['floor', 'table']
view = ['top', 'bottom']

random.seed(24)
random_list = list(range(1,14))

seq_name = []
for i in random_list:
    seq = 'seq'+str(i).zfill(2)
    seq_name.append(seq)

for i in background:
    for j in view:
        for seq in seq_name:
            images = os.listdir(os.path.join(data_path,i,j,seq,'rgb'))
            for image in images:
                shutil.copyfile(os.path.join(data_path, i, j, seq,'rgb', image),
                                os.path.join(dest_path, 'images',image))
                
            masks = os.listdir(os.path.join(data_path,i,j,seq,'label'))
            for mask in masks:
                shutil.copyfile(os.path.join(data_path, i, j, seq,'label', mask),
                                os.path.join(dest_path, 'labels', mask))