#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: dataset.py 
@time: 2020/04/01
@contact: xcxie17@fudan.edu.cn 
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os, sys
    

class COVID19Dataset(Dataset):
    def __init__(self, patient_data, image_root, mask_root, num_sampled_slice = 16, stage='train', size=(256, 256)):
        super(COVID19Dataset, self).__init__()
        self.cases = patient_data['cases']
        if 'labels' in patient_data:
            self.labels = patient_data['labels']
        else:
            self.labels = [-1] * len(self.cases)
        self.num_sampled_slice = num_sampled_slice
        self.image_root = image_root
        self.mask_root = mask_root
        self.stage = stage
        self.size = size
        
    def batch_rotate(self, images, params={}):
        pass
    def batch_scale(self, images, params={}):
        pass
    def batch_flip(self, images, params={}):
        pass
    
    def batch_transform(self, images):
#         augmented, _ = aug.spatial_transforms.augment_spatial(
#             images, 
#             do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
#             do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
#             do_scale=True, scale=(.9, 1.1),
#             border_mode_data='constant', border_cval_data=cval,
#             order_data=3,
#             p_el_per_sample=0.5,
#             p_scale_per_sample=.5,
#             p_rot_per_sample=.5,
#             random_crop=False
#         )
#         return augmented
        return images
        
    def __getitem__(self, index):
        case = self.cases[index]
        label = int(self.labels[index])
        case_dir = os.path.join(self.image_root, case)
        mask_dir = os.path.join(self.mask_root, case)
        
        image_paths = sorted(os.listdir(case_dir))
        if len(image_paths) <= 50:
            sample_indexes = np.linspace(0, len(image_paths)-1, num=self.num_sampled_slice).astype(np.int)
        else:
            center = int(len(image_paths) // 2)
            left = center - 25
            right = left + 50
            sample_indexes = np.linspace(left, right, num=self.num_sampled_slice).astype(np.int)
        image_paths = [image_paths[index] for index in sample_indexes]
        mask_paths = sorted(os.listdir(mask_dir))
        mask_paths = [mask_paths[index] for index in sample_indexes]
        images = []
        for index in range(len(image_paths)):
#             print(image_paths, mask_paths)
            image = cv2.imread(os.path.join(case_dir, image_paths[index]))
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
            mask = cv2.imread(os.path.join(mask_dir, mask_paths[index]), 0)
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_CUBIC)
            image[mask < 200, :] = 0
            image = image.astype(np.float32)
            image /= 255.0
            images.append(image)
        if self.stage in ['train']:
            images = self.batch_transform(images)
        
        images = np.stack(images, axis=0)
        images = np.transpose(images, [0, 3, 1, 2])
        return torch.from_numpy(images), torch.tensor(label, dtype=torch.long), case
            
    def __len__(self):
        return len(self.cases)
      
