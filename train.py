#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: xiexiancheng 
@license: Apache Licence 
@file: train.py 
@time: 2020/04/01
@contact: xcxie17@fudan.edu.cn 
""" 
import tqdm
import os
import logging
import sys
import cv2
import json
import codecs
import numpy as np
import argparse
import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import COVID19Dataset
from model import COVIDNet
from logger import Logger
from metric import compute_metrics


def get_data(patient_list, patient_data):
    cases = []
    labels = []
    for patient in patient_list:
        patient_cases = patient_data[patient]['cases']
        patient_label = patient_data[patient]['label']
        for case in patient_cases:
            cases.append(case)
            labels.append(patient_label)
    return {'cases': cases, 'labels': labels}

def draw_roc_curve(fpr, tpr, auc):
    pass
    
    
def test():
    # 12 9 14 7 11
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="experiment name")
    parser.add_argument("--ckpt_path", type=str, help="checkpoint path")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    seed = 42
    batch_size = 4
    num_sampled_slice = 16
    n_feats = 2048
    torch.manual_seed(seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(seed)
    image_root = './data_test'
    mask_root = './data_segmentation_test'
    patient_data_path = './data_test/data_test.json'
    with codecs.open(patient_data_path, 'r', encoding='utf-8') as reader:
        patient_data = json.load(reader)
    model_output_dir = './exps/{}'.format(args.exp_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)
    model = COVIDNet(2, n_feats=n_feats)
    state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    test_data = COVID19Dataset(patient_data, image_root, mask_root, num_sampled_slice=num_sampled_slice, stage='test')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=4)
    with torch.no_grad():
        model.eval()
        result = {}
        targets = []
        preds_raw = []
        for sample in tqdm.tqdm(test_dataloader):
            batch_images, batch_labels, batch_cases = sample
            batch_images = batch_images.cuda()
            batch_labels = batch_labels.numpy()
            pred_logits = model(batch_images)
            pred_raws = F.softmax(pred_logits, dim=-1)
            pred_raws = pred_raws.cpu().numpy()
            for index, case in enumerate(batch_cases):
                result[case] = float(pred_raws[index][1])
                targets.append(int(batch_labels[index]))
                preds_raw.append(float(pred_raws[index][1]))
        fpr, tpr, threshold = roc_curve(targets, preds_raw)   ## 计算真正率和假正率
        roc_auc = auc(fpr, tpr)                               ## 计算auc的值
        np.savez(os.path.join(model_output_dir, 'result.npz'), fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        # auc = roc_auc_score(targets, preds_raw)
#         plt.figure()
#         plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {:0.4f}'.format(roc_auc))
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic Curve')
#         plt.legend(loc="lower right")
#         plt.savefig(os.path.join(model_output_dir, 'roc.png'))
        print('{} auc is {:4f}'.format(args.exp_name, roc_auc))
        with codecs.open(os.path.join(model_output_dir, 'result.txt'), 'w', encoding='utf-8') as writer:
            for key, value in result.items():
                writer.write('{}\t{}\n'.format(key, value))
        

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="experiment name")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    epoches = 20
    batch_size = 4
    num_sampled_slice = 16
    seed = 42
    log_step = 5
    n_feats = 2048
    folds = 5
    patient_data_path = './data_selected/data.json'
    image_root = './data_selected'
    mask_root = './data_segmentation'
    torch.manual_seed(seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(seed)
    
    with codecs.open(patient_data_path, 'r', encoding='utf-8') as reader:
        patient_data = json.load(reader)
    patient_list = list(patient_data.keys())
    random.shuffle(patient_list)
    
    kf = KFold(n_splits=folds)
    kf.get_n_splits(patient_list)
    k = 0
    logging.basicConfig(filename='./exps/{}/log.all'.format(args.exp_name), 
                            level=logging.INFO, 
                            filemode='w',
                            format='%(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    for train_patient_index, val_patient_index in kf.split(patient_list):
        train_patient_list = [patient_list[index] for index in train_patient_index]
        val_patient_list = [patient_list[index] for index in val_patient_index]
        train_data = get_data(train_patient_list, patient_data)
        val_data = get_data(val_patient_list, patient_data)
        train_dataset = COVID19Dataset(train_data, image_root, mask_root, num_sampled_slice=num_sampled_slice, stage='train')
        val_dataset = COVID19Dataset(val_data, image_root, mask_root, num_sampled_slice=num_sampled_slice, stage='val')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=False) 
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False)
        
        k += 1                                                       
        log_output_dir = './exps/{}/{}/logs'.format(args.exp_name, k)
        model_output_dir = './exps/{}/{}/models'.format(args.exp_name, k)
        if not os.path.isdir(log_output_dir):
            os.makedirs(log_output_dir, exist_ok=True)
        if not os.path.isdir(model_output_dir):
            os.makedirs(model_output_dir, exist_ok=True)
        
        logging.info('{}/{} folds cross validation train patients: {}, train_cases: {}/{}/{}, val patients: {}, val_cases: {}/{}/{}'
              .format(
                      k,
                      folds,
                      len(train_patient_index), 
                      len(train_dataset),
                      np.sum(train_data['labels']),
                      len(train_dataset)-np.sum(train_data['labels']),
                      len(val_patient_index),
                      len(val_dataset),
                      np.sum(val_data['labels']),
                      len(val_dataset)-np.sum(val_data['labels']),
                     ))
        model = COVIDNet(2, n_feats=n_feats)
        optimizer = optim.Adam(model.parameters(), lr=4e-4)
        criterion = nn.CrossEntropyLoss()
        model = model.cuda()
        summary_writer = Logger(log_output_dir)

        for epoch in range(epoches):
            model.train()
            epoch_loss = 0.0
            print('*'*40)
            for step, sample in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_images, batch_labels, batch_cases = sample
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()
                pred_logits = model(batch_images)
                
                pred_raws = F.softmax(pred_logits, dim=-1)
                pred_labels = torch.argmax(pred_raws, dim=1)
                pred_raws = pred_raws[:, 1]
                loss = criterion(pred_logits, batch_labels)
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.25)
                optimizer.step()
                _loss = loss.item()
                epoch_loss += _loss
                pred_labels = pred_labels.detach().cpu().numpy()
                pred_raws = pred_raws.detach().cpu().numpy()
                _targets = batch_labels.cpu().numpy()
                if step % log_step == 0 or step == len(train_dataloader) - 1:
                    message = 'epoch {} step {}/{} loss is {:.6f}/{:.6f}'.format(epoch, step, len(train_dataloader),  _loss, epoch_loss / (step + 1))
#                     metrics = compute_metrics(pred_labels, _targets, preds_raw=pred_raws)
#                     for k,v in metrics.items():
#                         message+='{}:{} '.format(k, v)
                    logging.info(message)
            model_path = os.path.join(model_output_dir, 'covidnet_{}.pth'.format(epoch))
            info_to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
                'epoch':epoch
            }
            torch.save(info_to_save, model_path)
            
            # evaluation
            val_pred_raws = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                model.eval()
                for step, sample in enumerate(val_dataloader):
                    batch_images, batch_labels, batch_cases = sample
                    batch_images = batch_images.cuda()
                    pred_logits = model(batch_images)
                    pred_raws = F.softmax(pred_logits, dim=-1)
                    pred_labels = torch.argmax(pred_raws, dim=1)
                    pred_raws = pred_raws[:, 1]
                    
                    pred_raws = pred_raws.cpu().numpy()
                    pred_labels = pred_labels.cpu().numpy()
                    batch_labels = batch_labels.numpy()
                    if len(val_pred_raws):
                        val_pred_raws = np.concatenate((val_pred_raws, pred_raws), axis=0)
                        val_preds = np.concatenate((val_preds, pred_labels), axis=0)
                        val_targets = np.concatenate((val_targets, batch_labels), axis=0)     
                    else:
                        val_pred_raws = pred_raws
                        val_preds = pred_labels
                        val_targets = batch_labels
                print(val_preds, val_targets, val_pred_raws)
                metrics = compute_metrics(val_preds, val_targets, preds_raw=val_pred_raws)
                message = ''
                for key, value in metrics.items():
                    message+='{}:{} '.format(key, value)
                    summary_writer.add_scalar_summary('val/{}'.format(key), value, epoch)
                logging.info(message)
           
                    
if __name__ == '__main__':
    # train()
    test()