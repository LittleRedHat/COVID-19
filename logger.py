#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: logger.py 
@time: 2020/04/01
@contact: xcxie17@fudan.edu.cn 
"""
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar_summary(self, key, value, iteration):
        if self.writer:
            self.writer.add_scalar(key, value, iteration)

    def add_image_summary(self, tag, image_tensor, iteration):
        if self.writer:
            self.writer.add_image(tag, image_tensor, iteration)
