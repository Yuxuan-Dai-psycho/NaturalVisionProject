import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler 


#%% Scale diff
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
# roi
select_network = [1, 2, 10, 11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
print(f'Select {voxel_selected.sum()} voxels')
data = data_raw[class_loc, :roi.shape[1]]
data = data[:, voxel_selected]
run_idx = run_idx[class_loc]
label = label[class_loc]
del data_raw


methods = ['M2', 'M2+M1', 'No_scale']
for method in methods:
    print(f'Start scale in {method}')
    # preprocess for scaling
    scaler = StandardScaler()
    data_scale = np.zeros(data.shape)
    for idx in range(n_run):
        tmp_data = data[run_idx==idx, :]
        if method == 'M2' or method == 'M2+M1':
            # scale on row, based on all run info
            for idx_scale in range(tmp_data.shape[0]):
                tmp_sample = tmp_data[idx_scale]
                tmp_data[idx_scale] = (tmp_sample - np.mean(tmp_sample)) / np.std(tmp_sample)
            print('Finish row scaling')
            if method == 'M2+M1':
                tmp_data = scaler.fit_transform(tmp_data)
                print('Finish column scaling')
        # scale on column, based on feature
        elif method =='M1' :
            tmp_data = scaler.fit_transform(tmp_data)
            print('Finish column scaling')
        elif method =='No_scale':
            pass
        # fill data
        data_scale[run_idx==idx, :] = tmp_data

    # label and run_idx
    label_all = np.zeros((data_scale.shape[0], 2))
    label_all[:, 0] = label
    label_all[:, 1] = run_idx
    
    # save data
    np.save(pjoin(main_path, 'imagenet_decoding', 'scale', f'{sub_name}_imagenet-response_{method}.npy'), data_scale)
    np.save(pjoin(main_path, 'imagenet_decoding', 'scale', f'{sub_name}_imagenet-label&run_idx_{method}.npy'), label_all)

    # # construct mean pattern
    n_class = np.unique(label).shape[0]
    n_compare = pd.DataFrame(label).value_counts().min()
    n_loop = 1000
    
    class_pattern = np.zeros((n_loop, n_class, data_scale.shape[1]))
    # we will do random selection here
    # to minimize the randomness, loop 1000 times here
    for loop_idx in range(n_loop):
        for idx,class_idx in enumerate(np.unique(label)):
            class_loc = label == class_idx 
            # random select sample to make each class has the same number
            data_sample = data_scale[class_loc, :][np.random.choice(np.sum(class_loc), n_compare, replace=False)]
            class_pattern[loop_idx, idx, :] = np.mean(data_sample, axis=0)
        print('Finish %04d/%04d random selection'%(loop_idx+1, n_loop))
    class_pattern = np.mean(class_pattern, axis=0)
    
    np.save(pjoin(main_path, 'imagenet_decoding', 'scale', f'{sub_name}_imagenet-class_pattern_{method}.npy'), class_pattern)

