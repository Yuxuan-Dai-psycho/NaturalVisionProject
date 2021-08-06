import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler 



#%% roi level

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding')

n_run = 40
run_idx = np.repeat(np.linspace(0,n_run-1,n_run, dtype=int), 100)

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)


sub_id = 2
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)
data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-beta.mat')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-label.npy')

# data, label and run_idx
data_raw = sio.loadmat(data_path)['response']
label = np.load(label_path)

# define select class and roi
# class_selected = [1, 6, 8, 9, 12, 14, 16, 17, 
#                   18, 19, 20, 21, 22, 24, 25, 26]
class_selected = [1, 6, 8, 9, 16, 18, 19, 
                  20, 24, 25]
class_loc = np.asarray([True if x in class_selected else False for x in label])

run_idx = run_idx[class_loc]
label = label[class_loc]

# label and run_idx
label_all = np.zeros((label.shape[0], 2))
label_all[:, 0] = label
label_all[:, 1] = run_idx
np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
              f'{sub_name}_imagenet-label-num_prior.npy'), label_all)

# roi
roi_all = ['V1','V2','V3','V4','V8',
           'FFC','VVC','VMV1','VMV2','VMV3']

for roi_name in roi_all:
    # get roi
    tmp_index = roi_names[roi_names.iloc[:,0]==f'L_{roi_name}_ROI'].index.tolist()[0]+1
    roi_index = [tmp_index, tmp_index+180] #plus 180 for right brain
    voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
    print(f'Select {voxel_selected.sum()} voxels in {roi_name}')
    data_tmp = data_raw[class_loc, :roi.shape[1]]
    data = data_tmp[:, voxel_selected]

    # preprocess for scaling
    scaler = StandardScaler()
    data_scale = np.zeros(data.shape)
    for idx in range(n_run):
        tmp_data = data[run_idx==idx, :]
        data_scale[run_idx==idx, :] = scaler.fit_transform(tmp_data)

    # save data
    np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
                  f'{sub_name}_imagenet-response_{roi_name}.npy'), data_scale)

    # # construct mean pattern
    n_class = np.unique(label).shape[0]
    n_compare = pd.DataFrame(label).value_counts().min()
    n_loop = 1000
    
    class_pattern = np.zeros((n_loop, n_class, data_scale.shape[1]))
    # we will do random selection here
    # to minimize the randomness, loop 1000 times here
    for loop_idx in range(n_loop):
        for idx,class_idx in enumerate(np.unique(label)):
            tmp_class_loc = label == class_idx 
            # random select sample to make each class has the same number
            data_sample = data_scale[tmp_class_loc, :][np.random.choice(np.sum(tmp_class_loc), 
                                                                        n_compare, replace=False)]
            class_pattern[loop_idx, idx, :] = np.mean(data_sample, axis=0)
        print('Finish %04d/%04d random selection'%(loop_idx+1, n_loop))
    class_pattern = np.mean(class_pattern, axis=0)
    
    np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
                  f'{sub_name}_imagenet-class_pattern_{roi_name}.npy'), class_pattern)


#%% network level
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding')

n_run = 40
run_idx = np.repeat(np.linspace(0,n_run-1,n_run, dtype=int), 100)

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

    
sub_id = 3
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)
data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-beta.mat')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-label.npy')

# data, label and run_idx
data_raw = sio.loadmat(data_path)['response']
label = np.load(label_path)

# define select class and roi
# class_selected = [1, 6, 8, 9, 12, 14, 16, 17, 
#                   18, 19, 20, 21, 22, 24, 25, 26]
class_selected = [1, 6, 8, 9, 12, 17, 
                  21, 22, 24, 25]
class_loc = np.asarray([True if x in class_selected else False for x in label])

run_idx = run_idx[class_loc]
label = label[class_loc]

# label and run_idx
label_all = np.zeros((label.shape[0], 2))
label_all[:, 0] = label
label_all[:, 1] = run_idx
np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
              f'{sub_name}_imagenet-label&run_idx.npy'), label_all)

# roi
roi_all = {'visual1':[1],
           'visual2':[2],
           'multi_posterior':[10],
           'multi_ventral':[11],
           'all':[1, 2, 10, 11]}

for network_name in roi_all.keys():
    select_network = roi_all[network_name]
    # get roi
    roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
    voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
    print(f'Select {voxel_selected.sum()} voxels in {network_name}')
    data_tmp = data_raw[class_loc, :roi.shape[1]]
    data = data_tmp[:, voxel_selected]

    # preprocess for scaling
    scaler = StandardScaler()
    data_scale = np.zeros(data.shape)
    for idx in range(n_run):
        tmp_data = data[run_idx==idx, :]
        data_scale[run_idx==idx, :] = scaler.fit_transform(tmp_data)

    # save data
    np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
                  f'{sub_name}_imagenet-response_{network_name}.npy'), data_scale)

    # # construct mean pattern
    n_class = np.unique(label).shape[0]
    n_compare = pd.DataFrame(label).value_counts().min()
    n_loop = 1000
    
    class_pattern = np.zeros((n_loop, n_class, data_scale.shape[1]))
    # we will do random selection here
    # to minimize the randomness, loop 1000 times here
    for loop_idx in range(n_loop):
        for idx,class_idx in enumerate(np.unique(label)):
            tmp_class_loc = label == class_idx 
            # random select sample to make each class has the same number
            data_sample = data_scale[tmp_class_loc, :][np.random.choice(np.sum(tmp_class_loc), 
                                                                        n_compare, replace=False)]
            class_pattern[loop_idx, idx, :] = np.mean(data_sample, axis=0)
        print('Finish %04d/%04d random selection'%(loop_idx+1, n_loop))
    class_pattern = np.mean(class_pattern, axis=0)
    
    np.save(pjoin(main_path, 'imagenet_decoding', 'roi', 
                  f'{sub_name}_imagenet-class_pattern_{network_name}.npy'), class_pattern)

#%%
import numpy as np
import pandas as pd
from os.path import join as pjoin

from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_selected = [1, 6, 8, 9, 12, 17, 
                  21, 22, 24, 25]
n_class = len(class_selected)

# get class sample
class_mapping = pd.read_csv(pjoin(main_path, 'superClassMapping.csv'))
super_class_name, super_class_number = [], []
for i in range(30):
    superclass_df = class_mapping.loc[class_mapping['superClassID']==i+1, 'superClassName']
    super_class_name.append("%d.%s" % (i+1, superclass_df.values[0]))
    super_class_number.append(len(superclass_df))
labels_selected = np.array(super_class_name)[np.array(class_selected)-1]

labels = ['vis1', 'vis2', 'pmm', 'vmm', 'visual_all']
eucl_distance = np.zeros((n_class, len(labels)))
for idx,roi_name in enumerate(labels):
    print(roi_name)
    file_name = f'sub-03_imagenet-class_pattern_{roi_name}.npy'
    # get Euclidean distance
    class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', 'roi', file_name))
    eucl_matrix =  distance_matrix(class_pattern, class_pattern)
    eucl_distance[:, idx] = np.mean(eucl_matrix, axis=0)

x_1 = 5*np.arange(n_class)
x_2 = x_1 - 1.6
x_3 = x_1 - 0.8
x_4 = x_1 + 0.8
x_5 = x_1 + 1.6


# for all sample
plt.figure(figsize=(20,12))
plt.bar(x_1, eucl_distance[:, 0], label='vis1', color='#F54748')
plt.bar(x_2, eucl_distance[:, 1], label='vis2', color='#0A81AB')
plt.bar(x_3, eucl_distance[:, 2], label='pmm', color='#7952B3')
plt.bar(x_4, eucl_distance[:, 3], label='vmm', color='#4D2C37')
plt.bar(x_5, eucl_distance[:, 4], label='visual_all', color='#9E8C8A')
# plt.bar(x_4, eucl_distance[:, 3], label='No_scale', color='#0A81AB')

font_title = {'family': 'arial', 'weight': 'bold', 'size':20}
font_other = {'family': 'arial', 'weight': 'bold', 'size':16}
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_ylim(0,25)

plt.xticks((x_1 + x_2 + x_3 + x_4 + x_5)/5, labels_selected, rotation=45,
           fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)
plt.legend(prop=font_other)

plt.ylabel('distance', font_other)
plt.title('Euclidean distance in different ROIs', font_title)
plt.savefig(pjoin(out_path, 'distance_bar_roi.jpg'), bbox_inches='tight')
plt.close()


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

