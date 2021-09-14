#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:21:34 2021

@author: gongzhengxin
"""
import time, os, pickle, json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from random import choice
from os.path import join as pjoin
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
def run_based_transform(data, info):
    """
    
    """
    std = StandardScaler()
    run_idxes = np.unique(info)
    new_data = np.zeros_like(data)
    for run_idx in run_idxes:
        slice = (info==run_idx)

        new_data[slice, :] = std.fit_transform(data[slice, :])

    return new_data

def mean_pattern(data, info):
    """
    info: label info
    """
    unique_label = np.unique(info)
    new_data = np.vstack(tuple([data[info==_, :].mean(axis=0) for _ in unique_label]))
    return new_data

def network_selection(network_index):
    """
    parameters
    ----------
    network_indx : int or list
    """
    main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
    network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
    roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
    roi = sio.loadmat(roi_path)['glasser_MMP']
    network = sio.loadmat(network_path)['netassignments']
    network = [x[0] for x in network]

    # Select roi that in visual area
    if type(network_index) == int:
        select_network = [network_index]
    elif type(network_index) != list:
        raise AssertionError('Augment error: network_index should be integer or list of interger!')
    else:
        select_network = network_index
    roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
    return np.asarray([True if x in roi_index else False for x in roi[0]])

class SaveOut(BaseEstimator, TransformerMixin):
    def transform(self, X):
        shape = X.shape
        self.shape = shape
        self.data = X
        # what other output you want
        return self

    def fit(self, X, y=None, **fit_params):
        return self

subjects  = ['sub-core02', 'sub-core03']
data_path = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results'

data, data_info = [], []
for sub in subjects:
    data.append(np.load(pjoin(data_path, 'imagenet_decoding', '_'.join([sub, '14class-cortex.npy'])))) 
    data_info.append(np.load(pjoin(data_path, 'imagenet_decoding', '_'.join([sub, '14class-label-runidx.npy']))))

# standardize
for idx in [0,1]:
    data[idx] = run_based_transform(data[idx], data_info[idx][:,1])


# mean pattern transformation
mean_data, mean_data_info = [], []
for sub_idx in [0,1]:
    run_idex = np.unique(data_info[sub_idx][:,1])
    cur_data = [mean_pattern(data[sub_idx][data_info[sub_idx][:,1]==_, :], data_info[sub_idx][data_info[sub_idx][:,1]==_, 0]) for _ in run_idex]
    mean_data.append(np.vstack(tuple(cur_data)))
    new_label = np.hstack([np.unique(data_info[sub_idx][data_info[sub_idx][:,1]==_, 0]) for _ in run_idex])
    new_runidx = []
    for _ in run_idex:
        new_runidx.extend([_]*len(np.unique(data_info[sub_idx][data_info[sub_idx][:,1]==_, 0])))
    new_info = np.vstack((new_label, np.array(new_runidx))).transpose()
    mean_data_info.append(new_info)

data, data_info = mean_data, mean_data_info

# animation transformation
animate_dict = {1:0,  12:1, 14:1, 16:1, 17:0, 18:0, 19:0, 20:0, 21:0, 24:0, 25:0, 6:1, 8:1, 9:1}
for subid in [0,1]:
    for i in range(len(data_info[subid])):
        data_info[subid][i,0] = animate_dict[data_info[subid][i,0]]


# primary visual: [1]
# secondary visual: [2]
# posterior multimodal: [10]
# ventral multimodal: [11]('preprocess', StandardScaler()), 
pipe = Pipeline([('visualization', TSNE(random_state=42))])
colors = ['#687980', '#fbc6a4', '#8e9775',  '#afb9c8', '#233e8b', '#1eae98', '#94d0cc', 
          '#9fe6a0', '#4aa96c', '#f4a9a8',  '#34656d', '#00adb5', '#ce97b0', '#8ab6d6' ]
nets = {'PrimaryVisual':[1], 'SecondaryVisual': [2],
        'PosteriorMultimodal': [10], 'VentralMultimodal': [11],
        'VisualAll':[1,2,10,11]}
print('===========================================')
for area, Aindex in nets.items():
    print('    current area:   ', area)
    network_mask = network_selection(Aindex)
    for sub, index in zip(subjects, [0,1]):
        print('     current subject:   ', sub)
        X_embedded = pipe.fit_transform(data[index][:, network_mask])
        unique_label = np.unique((data_info[0][:,0]))
        unique_runidx = np.unique((data_info[0][:,1]))
        plt.figure(figsize=(10, 10))
        plt.xlim(X_embedded[:, 0].min(), X_embedded[:, 0].max() + 1)
        plt.ylim(X_embedded[:, 1].min(), X_embedded[:, 1].max() + 1)
        for i in range(len(data_info[index])):
            # actually plot the digits as text instead of using scatter
            plt.text(X_embedded[i, 0], X_embedded[i, 1], str(data_info[index][i,0]),
            color = colors[np.where(unique_label==data_info[index][i,0])[0][-1]],
            fontdict={'weight': 'bold', 'size': 9})
        plt.savefig(pjoin(data_path, 'imagenet_decoding', 'results', '_'.join([sub, area, 'mean.png'])))
        print('save out : ', pjoin(data_path, 'imagenet_decoding', 'results', '_'.join([sub, area, 'mean.png'])))
        plt.figure(figsize=(10, 10))
        plt.xlim(X_embedded[:, 0].min(), X_embedded[:, 0].max() + 1)
        plt.ylim(X_embedded[:, 1].min(), X_embedded[:, 1].max() + 1)
        for i in range(len(data_info[index])):
            # actually plot the digits as text instead of using scatter
            plt.text(X_embedded[i, 0], X_embedded[i, 1], str(data_info[index][i,1]),
            color = colors[np.where(unique_runidx==data_info[index][i,1])[0][-1]%14],
            fontdict={'weight': 'bold', 'size': 9}) #int((np.where(unique_runidx==data_info[index][i,1])[0][-1])/10)
        plt.savefig(pjoin(data_path, 'imagenet_decoding', 'results', '_'.join([sub, area, 'mean_runidx.png'])))
        print('save out : ', pjoin(data_path, 'imagenet_decoding', 'results', '_'.join([sub, area, 'mean_runidx.png'])))
