import nibabel as nib
import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
import os, pickle

#%% load data and roi mat
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'

data_path = pjoin(main_path, f'imagenet_decoding/{sub_name}_imagenet-response.mat')
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')

data_raw = sio.loadmat(data_path)['response']
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

#%% transform data into roi organization
n_roi = 180
data = data_raw[:, :roi.shape[1]]

data_roi_mean = np.zeros((data_raw.shape[0], n_roi))
data_roi_all = dict()

for roi_idx in range(n_roi):
    # get data
    voxel_idx = np.where((roi==roi_idx+1) | (roi==roi_idx+181))[1] # roi_idx, roi_idx+180 corresponds to L, R
    data_roi = data[:, voxel_idx]
    # store data
    roi_name = roi_names.iloc[roi_idx,0].split('_')[1]
    data_roi_all[roi_name] = data_roi
    data_roi_mean[:, roi_idx] = np.mean(data_roi, axis=1)
    print(f'Finish transforming data in ROI: {roi_name}')

# save data
np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet_roi-mean-response.npy'), data_roi_mean)
with open(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet_roi-response.pkl'),'wb') as f:
    pickle.dump(data_roi_all, f)



