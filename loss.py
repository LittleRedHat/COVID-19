#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: loss.py 
@time: 2020/04/16
@contact: xcxie17@fudan.edu.cn 
"""
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, eps=1e-8, gamma=2.0, positive_weight=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.positive_weight = positive_weight
        self.reduction = reduction

    def forward(self, targets, preds):
        preds = torch.clamp(preds,  self.eps, 1 - self.eps)
        loss = - (self.positive_weight * torch.pow(1 - preds, self.gamma) * targets * torch.log(preds)
                  + torch.pow(preds, self.gamma) * (1 - targets) * torch.log(1 - preds))
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss