import nibabel as nib
import time, os, pickle, json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from random import choice
from os.path import join as pjoin
roi_mat = None
ciftify_dir = '/nfs/s2/userhome/gongzhengxin/nvd/NaturalObject/link_to_ciftify'

save_path = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/'
nifti_dir = '/nfs/s2/userhome/gongzhengxin/nvd/NaturalObject/data/bold/.nifti_old'

result_dir = 'MNINonLinear/.Results/'
# --20210331 the following code forget to consider the actual sequence of experiment 
# --20210406 rewrite to fix the problem, reveal that missing .sort() casue the sequence before was messing
# input dirs  
def save_npy():
    for sub_dir in ['sub-core02', 'sub-core03']:
        # MNINolinear/Results disposit all the runs data
        _result_path = pjoin(ciftify_dir, sub_dir, result_dir)
        # extract the ImageNet runs
        imagenet_runs = [_ for _ in os.listdir(_result_path) if ('ImageNet' in _) and ('discard' not in _)]
        imagenet_runs.sort() # sort() to be [1 10 2 -- 9]
        # initialize the mapping dict
        stim_resp_map = {}
        # loop run
        for single_run in imagenet_runs:
            print(single_run)
            # collect session number & run number
            ses = int((single_run.split('_')[0]).replace('ses-ImageNet', ''))
            runidx = int(single_run.split('_')[2].replace('run-', ''))
            # prepare .feat/GrayordinatesStats dir
            nii_dir = '{0}'.format(single_run)
            nii_path = pjoin(ciftify_dir, sub_dir, result_dir, nii_dir)
            # loop trial 
            
            nii_file = pjoin(nii_path, '{0}_Atlas.dtseries.nii'.format(single_run))
            dt_data = nib.load(nii_file).get_fdata()
            if roi_mat:
                # only save roi
                stim_resp_map[single_run] = np.array(dt_data[:,roi_mat])
            else:
                stim_resp_map[single_run] = np.array(dt_data)
        # transfer to matrix
        imagenet_data = np.dstack(tuple([stim_resp_map[_] for _ in list(stim_resp_map.keys())]))
        imagenet_data = imagenet_data.transpose((2,1,0))
        np.save('/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/{}_imagenet-nii-raw.npy'.format(sub_dir.replace('core','')), imagenet_data)


result_dir = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results'
sub_denoised = np.load(pjoin(result_dir, 'sub-03_imagenet-nii-denoised.npy'))
sub_raw = np.load(pjoin(result_dir, 'sub-03_imagenet-nii-raw.npy'))



def tsnr(data):
    mean = data.mean(axis=-1)
    std = data.std(axis=-1)
    return np.nan_to_num(mean/std)

plt.figure(figsize=(20,32))
tsnr_denoised, tsnr_raw = np.zeros_like(sub_denoised[:,:,0]), np.zeros_like(sub_raw[:,:,0])
for run in range(40):
    denoised_data = sub_denoised[run,:,:]
    raw_data = sub_raw[run,:,:]
    # 
    snr_den = tsnr(denoised_data)
    snr_raw = tsnr(raw_data)

    tsnr_denoised[run, :] = snr_den
    tsnr_raw[run, :] = snr_raw
 
    plt.subplot(8,5,run+1)
    plt.hist(snr_raw, color='blue', bins=100, alpha=0.6, label='raw')
    plt.hist(snr_den,  color='red', bins=100, alpha=0.6, label='denoised')
    plt.legend()
plt.show()

# np.save(pjoin(result_dir, 'sub-02tsnr-raw.npy'), tsnr_raw)
# np.save(pjoin(result_dir, 'sub-02_tsnr-den.npy'), tsnr_denoised)

plt.figure(figsize=(20,32))
tsnr_denoised, tsnr_raw = np.load(pjoin(result_dir, 'sub-03_tsnr-den.npy')), np.load(pjoin(result_dir, 'sub-03_tsnr-raw.npy'))
for run in range(40):

    snr_den = tsnr_denoised[run, :]
    snr_raw = tsnr_raw[run, :]

    plt.subplot(8,5,run+1)
    plt.hist(snr_raw[:59412], bins=100, color='blue', alpha=0.6, label='raw')
    plt.hist(snr_den[:59412],  bins=100, color='red', alpha=0.6, label='denoised')
    plt.legend()
plt.show()

tsnr_denoised_mean = tsnr_denoised.mean(axis=0)
tsnr_raw_mean = tsnr_raw.mean(axis=0)

plt.hist(tsnr_raw_mean[:59412], bins=200, color='blue', alpha=0.6, label='raw data')
plt.hist(tsnr_denoised_mean[:59412],  bins=200, color='red', alpha=0.6, label='denoised')
plt.legend()
plt.yticks([200,600,1000], [2,6,10])
plt.xlabel('tsnr (a.u.)')
plt.ylabel('#voxel number ($10^{2}$)')
plt.show()


def save_ciftifile(data, filename, template):
    ex_cii = nib.load(template)
    ex_cii.header.get_index_map(0).number_of_series_points = 1
    nib.save(nib.Cifti2Image(data.reshape((1,91282)),ex_cii.header), filename)


template = '/nfs/m1/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/sub-core03/MNINonLinear/\
Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'
#filename = pjoin(result_dir, 'sub03_tsnr_diff.dtseries.nii')
# save_ciftifile((tsnr_denoised_mean-tsnr_raw_mean), filename, template)

data_path = '/nfs/s2/userhome/gongzhengxin/nvd/Analysis_results/imagenet_decoding'
rois = ['vis1', 'vis2', 'vmm', 'pmm']
for roi in rois:
    filename = pjoin(data_path,f'sub-03_imagenet-{roi}.dtseries.nii')
    voxel_idx = np.load(pjoin(data_path, f'sub-03_imagenet-{roi}_voxel.npy'))
    data_visual_selection = np.zeros(91282)
    data_visual_selection[voxel_idx] = 1
    save_ciftifile(data_visual_selection, filename, template)
