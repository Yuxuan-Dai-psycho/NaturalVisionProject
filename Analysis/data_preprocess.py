import numpy as np
from os.path import join as pjoin

import pandas as pd
from os.path import join as pjoin
import scipy.io as sio


#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'

for sub_name in ['sub-core03',  'sub-core02']:
    data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_14class-cortex.npy')
    label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_14class-label-runidx.npy')
    out_path = pjoin(main_path, 'imagenet_decoding')
    
    # data, label and run_idx
    data_raw = np.load(data_path)
    label_raw = np.load(label_path)
    
    label = label_raw[:, 0]
    run_idx = label_raw[:, 1]
    
    # define select class
    class_delete = [12,14,17,20]
    class_selected = np.asarray([True if x not in class_delete else False for x in label])
    
    
    label_new = label_raw[class_selected, :]
    data_new = data_raw[class_selected, :]
    
    np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_10class-cortex.npy'), data_new)
    np.save(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_10class-label-runidx.npy'), label_new)

#%% select class based on distance


main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core03'

network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-response.mat')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

data_raw = sio.loadmat(data_path)['response']
label = np.load(label_path)

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

select_network = [1,2,10,11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
data = data_raw[:, :roi.shape[1]]
data = data[:, voxel_selected]
print(f'Select {voxel_selected.sum()} voxels')
del data_raw

# construct mean pattern
n_class = 30
class_pattern = np.zeros((n_class, data.shape[1]))
for class_idx in range(n_class):
    class_loc = label == class_idx + 1
    class_pattern[class_idx, :] = np.mean(data[class_loc, :], axis=0)
del data
    
#%% construct distance matrix

import matplotlib.pyplot as plt
import matplotlib.font_manager

# correlation distance
corr_matrix = np.corrcoef(class_pattern)
# corr_matrix[range(n_class), range(n_class)] = 0

plt.figure(figsize=(10,8))
cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(corr_matrix, cmap=cmap, norm=norm)
plt.colorbar()

plt.xlabel('Class idx', font)
plt.ylabel('Class idx', font)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), 
           np.linspace(1, n_class, n_class, dtype=np.uint8), 
            fontproperties='arial', weight='bold', size=10)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), 
           np.linspace(1, n_class, n_class, dtype=np.uint8), 
            fontproperties='arial', weight='bold', size=10)
plt.title('Distance measured by pearson correlation', font)
plt.savefig(pjoin(out_path, 'corr_RDM.jpg'))
plt.close()
    

corr_matrix[range(n_class), range(n_class)] = 0
corr_distance = np.mean(corr_matrix, axis=0)
print('Distant Order by correlation: ', np.argsort(corr_distance)+1)

#%% Euclidean distance

from scipy.spatial import distance_matrix

eucl_matrix =  distance_matrix(class_pattern, class_pattern)


plt.figure(figsize=(10,8))
cmap = plt.cm.jet
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(eucl_matrix, cmap=cmap)
plt.colorbar()

plt.xlabel('Class idx', font)
plt.ylabel('Class idx', font)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), 
           np.linspace(1, n_class, n_class, dtype=np.uint8), 
            fontproperties='arial', weight='bold', size=10)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), 
           np.linspace(1, n_class, n_class, dtype=np.uint8), 
            fontproperties='arial', weight='bold', size=10)
plt.title('Distance measured by Euclidean', font)
plt.savefig(pjoin(out_path, 'eucl_RDM.jpg'))
plt.close()
    

eucl_distance = np.mean(eucl_matrix, axis=0)
print('Distant Order by Euclidean: ', np.argsort(-eucl_distance)+1)





