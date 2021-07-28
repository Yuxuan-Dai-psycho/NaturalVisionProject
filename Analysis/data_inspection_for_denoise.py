#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 00:03:59 2021
2021/07/17 compare the raw beta and denoides data in
    fold:: /nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/imagenet_decoding
    compare the raw nii times series with denoised in
    fold:: /nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/snr_related
@author: gongzhengxin
"""
import os
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
import h5py
import pickle
import matplotlib.pyplot as plt

root_path = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/imagenet_decoding'

files = [_ for _ in os.listdir(root_path) if 'beta' in _]
files.sort()
os.chdir(root_path)

pkl_files = [_ for _ in os.listdir(root_path) if '.pkl' in _ and 'net-res' in _]
pkl_files.sort()

with open(pkl_files[0], 'rb') as f:
    pkl_raw = pickle.load(f)
with open(pkl_files[1], 'rb') as f:
    pkl_denoised = pickle.load(f)
del pkl_raw, pkl_denoised

betas = [_['response'] for _ in list(map(sio.loadmat, files[0:2]))]
# % cortex: 59412

plt.imshow(betas[0])
plt.show()

X = betas[0][0,:29706]
Y = betas[1][0,:29706]
plt.scatter(X, Y,color='blue',\
            edgecolor='white',alpha=0.8, s=25)
plt.plot([0,0],[Y.min(),Y.max()],ls='--',color='black')
plt.plot([X.min(),X.max()],[0,0],ls='--',color='black')
plt.plot([X.min(),X.max()],[X.min(),X.max()],color='black')
plt.xlabel('raw beta')
plt.ylabel('denoised')
plt.axis('equal')
plt.show()

# carpet plot
top4k_var_vox_idx = np.argsort(betas[1][:,0:29706].std(axis=0))[::-1][:4000]

plt.figure(figsize=(20,20))
plt.imshow(betas[1][:,top4k_var_vox_idx],vmax=90, vmin=-90)
plt.colorbar(shrink=0.6)
plt.show()

Ra_raw = betas[0][:,:29706].max(axis=1) - betas[0][:,:29706].min(axis=1)
Ra_den = betas[1][:,:29706].max(axis=1) - betas[1][:,:29706].min(axis=1)
Ra_ratio = Ra_den/Ra_raw
ratio = np.sum(Ra_ratio<1)/len(Ra_ratio)

# %%
root_path = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/snr_related'
os.chdir(root_path)
nii_raw = np.load('sub-03_imagenet-nii-raw.npy').astype(np.float16)
nii_den = np.load('sub-03_imagenet-nii-denoised.npy').astype(np.float16)


randindx = np.random.randint(0,29706,40)

plt.figure(figsize=(20,20))
for i,idx in enumerate(randindx):
    plt.subplot(6,6,i+1)
    sample_raw = nii_raw[:,idx,:]
    sample_den = nii_den[:,idx,:]






