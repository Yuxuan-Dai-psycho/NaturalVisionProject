#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 20:16:55 2021

@author: wenyushan
"""

# try data_pre fcn
#%% load data and label
import numpy as np
from os.path import join as pjoin
import scipy.io as sio
import data_pre

path = '/nfs/e2/workingshop/swap/imagenet_decoding'
sub_name = 'sub-core02'

data_path = pjoin(path, f'{sub_name}_imagenet-response.mat')
label_path = pjoin(path, f'{sub_name}_imagenet-label.npy')

data = sio.loadmat(data_path)['response']
label_raw = np.load(label_path)

#%% define roi
import pandas as pd

df_ROIs = pd.read_csv(pjoin(path, 'VVA_ROIs.csv'))
atlas = sio.loadmat(pjoin(path, 'glasser_atlas.mat'))['glasser2016'].squeeze()

roi_index = df_ROIs['label'].tolist()
voxel_selected = np.asarray([True if x in roi_index else False for x in atlas])

data = data[:, voxel_selected]
print('Finish data loading')

#%% outlier
# lpot
data_pca, out_mark = data_pre.plot_outlier(data, 0.05)

# remove
data_pro = data_pre.remove_outlier(data_pca, out_mark)