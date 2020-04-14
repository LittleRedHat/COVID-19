#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: model.py 
@time: 2020/04/01
@contact: xcxie17@fudan.edu.cn 
"""
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import models


class COVIDNet(nn.Module):
    def __init__(self, n_classes, n_feats=2048, target_layer=[]):
        super().__init__()
        model = models.resnet50(pretrained=True)
        layer_list = list(model.children())[:-2]
        self.features = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(n_feats, n_classes)
        self.n_classes = n_classes
        self.target_layer = target_layer
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
        
    def forward(self, x, return_feats=False):
        batch_size = x.size(0)
        window_size = x.size(1)
        x = x.reshape(batch_size * window_size, x.size(2), x.size(3), x.size(4))
        features = self.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(batch_size, window_size, -1)
        fusion_features, _ = torch.max(pooled_features, dim=1)
        output = self.classifer(fusion_features)
        if return_feats:
            features.register_hook(self.save_gradient)
            return output, features
        else:
            return output
    
