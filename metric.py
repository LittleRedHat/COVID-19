#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: metric.py 
@time: 2020/04/01
@contact: xcxie17@fudan.edu.cn 
""" 
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np


def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def compute_metrics(preds, targets, preds_raw=None):
    
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, preds_raw)
    return {'auc': auc, 
            'f1': f1, 
            'acc': acc, 
            'precision': precision, 
            'recall': recall,
            'tn': tn,
            'fp': fp,
            'tp': tp,
            'fn': fn
           }
