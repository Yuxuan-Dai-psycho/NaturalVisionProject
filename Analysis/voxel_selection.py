#%% get select voxel idx

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import scipy.io as sio
import pandas as pd
from model_utils import class_sample
from sklearn.feature_selection import SelectPercentile

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# load data
sub_id = 3
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)
data_path = pjoin(main_path, 'imagenet_decoding', 'roi', f'{sub_name}_imagenet-response_all.npy')
label_path = pjoin(main_path, 'imagenet_decoding', 'roi', f'{sub_name}_imagenet-label&run_idx.npy')

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

select_network = [1, 2, 10, 11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
visual_all_idx = np.asarray([idx for idx,x in enumerate(roi[0]) if x in roi_index])

net_names = ['vis1', 'vis2', 'pmm', 'vmm']
for net_name,net in zip(net_names, select_network):
    roi_tmp_index = [idx+1 for idx,x in enumerate(network) if x == net]
    print(net_name + '\n')
    print(roi_names.iloc[roi_tmp_index[:int(len(roi_tmp_index)/2)], 0].tolist())
    tmp_idx = np.array([idx for idx,x in enumerate(roi[0]) if x in roi_tmp_index])
    np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-{net_name}_voxel.npy'), tmp_idx)
    
# data, label and run_idx
data = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]
data, label, run_idx = class_sample(data, label, run_idx)

# feature selection on data
feature_selection = SelectPercentile(percentile=25)
feature_selection.fit(data, label)
select_idx = feature_selection.get_support(indices=True)
visual_select_idx = visual_all_idx[select_idx]

np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-visual_all_voxel.npy'), visual_all_idx)
np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-visual_select_voxel.npy'), visual_select_idx)

roi_used = []
for voxel in visual_select_idx:
    roi_one_voxel = roi[0, voxel]
    if roi_one_voxel <= 180:
        tmp_name = roi_names.iloc[int(roi_one_voxel-1), 0]
        roi_used.append(tmp_name)
roi_used = np.unique(np.array(roi_used))


#%% get whole brain select idx

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import scipy.io as sio
import pandas as pd
from model_utils import class_sample
from sklearn.feature_selection import SelectPercentile

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# load data
sub_id = 2
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)
data_path = pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-beta-whole_brain.npy')
label_path = pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-label-num_prior.npy')

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)
    
# data, label and run_idx
data = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]
data, label, run_idx = class_sample(data, label, run_idx)

# feature selection on data
feature_selection = SelectPercentile(percentile=10)
feature_selection.fit(data, label)
select_idx = feature_selection.get_support(indices=True)

np.save(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-whole_brain_select_idx.npy'), 
        select_idx)

roi_used = []
for voxel in select_idx:
    roi_one_voxel = roi[0, voxel]
    if roi_one_voxel <= 180:
        tmp_name = roi_names.iloc[int(roi_one_voxel-1), 0]
        roi_used.append(tmp_name)
roi_used = np.unique(np.array(roi_used))

#%% get active-based and stability-based voxel idx
import numpy as np
from os.path import join as pjoin
import scipy.io as sio
import pandas as pd
from model_utils import class_sample
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import pairwise_distances

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# load data
sub_id = 3
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)
data_path = pjoin(main_path, 'imagenet_decoding', 'ica', f'{sub_name}_imagenet-beta.npy')
label_path = pjoin(main_path, 'imagenet_decoding','ica', f'{sub_name}_imagenet-label-num_prior.npy')

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

select_network = [1, 2, 10, 11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
visual_all_idx = np.asarray([idx for idx,x in enumerate(roi[0]) if x in roi_index])

# data, label and run_idx
data = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]
data, label, run_idx = class_sample(data, label, run_idx)

# feature selection on data active mode
feature_selection = 25
n_voxel_select = int(data.shape[1]*(feature_selection/100))

# active-based selection
active_pattern = np.max(data, axis=0)
active_loc = np.argsort(-active_pattern)[:n_voxel_select]
active_idx = visual_all_idx[active_loc]
np.save(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-active_idx.npy'), 
        active_idx)

# stability-based voxel idx
run_intere = []
for run_label in np.unique(run_idx):
    run_loc = run_idx == run_label
    label_run = label[run_loc]
    if np.unique(label_run).shape[0] == np.unique(label).shape[0]:
        run_intere.append(run_label)

stability_score = np.zeros((data.shape[1]))
for voxel_idx in range(data.shape[1]):
    data_voxel = data[:, voxel_idx]
    # define pattern for each voxel
    # find runs that have 10 class
    voxel_pattern = np.zeros((len(run_intere), np.unique(label).shape[0]))
    for run_loop_idx,run_label in enumerate(run_intere):
        run_loc = run_idx == run_label
        data_run = data_voxel[run_loc]
        label_run = label[run_loc]
        for class_loop_idx,class_label in enumerate(np.unique(label)):
            class_loc = label_run == class_label
            data_class = data_run[class_loc]
            voxel_pattern[run_loop_idx, class_loop_idx] = np.mean(data_class, axis=0)
    print(f'Finish computing {voxel_idx} voxels')
    # compute stability score
    corr_matrix = pairwise_distances(voxel_pattern, metric='correlation')
    valid_value = np.triu(corr_matrix, 1).flatten()
    stability_score[voxel_idx] = np.mean(valid_value[valid_value!=0])
        
stability_loc = np.argsort(stability_score)[:n_voxel_select]
stability_idx = visual_all_idx[stability_loc]
np.save(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-stability_idx.npy'), 
        stability_idx)
np.save(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-stability_score.npy'), 
        stability_score)

# preprocessing on voxel selection
# feature_selection = 25
# n_voxel_select = int(data.shape[1]*(feature_selection/100))
# stability_score = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', 
#                           f'{sub_name}_imagenet-stability_score.npy'))
# stability_loc = np.argsort(stability_score)[:n_voxel_select]
# data = data[:, stability_loc]
stability_idx = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-stability_idx.npy'))
discrim_idx = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', f'{sub_name}_imagenet-visual_select_voxel.npy'))

#%% prepare voxel and roi index info used in DNN 
import pickle as pkl

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding')

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

# roi
select_network = [1, 2, 10, 11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])

# preprare saving data
voxel_to_roi = roi.squeeze()[voxel_selected]
roi_to_network = [x for idx,x in enumerate(network) if x in select_network]


dnn_struct_info = {'voxel_to_roi': voxel_to_roi,
                   'roi_to_network': roi_to_network}

with open(pjoin(out_path, 'dnn_struct_info.pkl'), 'wb') as f:
    pkl.dump(dnn_struct_info, f)


#%% visualize voxel
import nibabel as nib
import numpy as np
from os.path import join as pjoin


def save_ciftifile(data, filename, template):
    ex_cii = nib.load(template)
    ex_cii.header.get_index_map(0).number_of_series_points = 1
    nib.save(nib.Cifti2Image(data.reshape((1,91282)),ex_cii.header), filename)

sub_id = 3
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)

data_path = '/nfs/m1/BrainImageNet/Analysis_results/imagenet_decoding/voxel'
selected_voxel = np.load(pjoin(data_path, f'{sub_name}_imagenet-stability_idx.npy'))

voxel_all = np.zeros(91282)
voxel_all[selected_voxel] = 1

template = f'/nfs/m1/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/{sub_core_name}/MNINonLinear/\
Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'

filename = pjoin(data_path, f'{sub_name}_imagenet-stability_idx.dtseries.nii')
save_ciftifile(voxel_all, filename, template)

