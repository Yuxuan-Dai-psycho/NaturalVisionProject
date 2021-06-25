import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio

from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# personal function
from model_utils import nested_cv, find_outlier, gen_param_grid

#%% define params
n_class = 30
n_run = 40
n_train = 30
n_test = 10

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core03'

network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_10class-cortex.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_10class-label-runidx.npy')
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# data, label and run_idx
data_raw = np.load(data_path)
label_raw = np.load(label_path)

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

#%% grid search loop
info = pd.DataFrame(columns=['single', 'mean'])
        
#% Select roi that in visual area
select_network = [1,2,10,11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
data = data_raw[:, voxel_selected]
print(f'Select {voxel_selected.sum()} voxels')
label = label_raw[:, 0]
run_idx = label_raw[:, 1]

#% =======Enhance data====================
# ======== remove outlier=================
out_index = find_outlier(data, label, 0.05)
data = np.delete(data, out_index, axis=0)
label = np.delete(label, out_index, axis=0)
run_idx = np.delete(run_idx, out_index, axis=0)

# ========preprocessing on each run======
method = ''
scaler = StandardScaler()
data_scale = np.zeros(data.shape)
for idx in range(n_run):
    tmp_data = data[run_idx==idx, :]
    # scale on row, based on all run info 
    mean_pattern = np.mean(tmp_data, axis=0)
    if method == 'norm':
        tmp_data = tmp_data/np.linalg.norm(mean_pattern)
    elif method == 'substract':
        for idx_scale in range(tmp_data.shape[0]):
            tmp_data[idx_scale] = (tmp_data[idx_scale] - np.mean(mean_pattern))/np.std(mean_pattern)
    else:
        pass 
    # scale on column, based on feature
    data_scale[run_idx==idx, :] = scaler.fit_transform(tmp_data)
        
# define dual or primal formulation
# dual = False if X_train.shape[0] > X_train.shape[1] else True
# =============== start grid searching8==============
# define params for grid search
param_grid = gen_param_grid('svm')                 

# make pipeline
pipe = Pipeline([('feature_selection', SelectPercentile()),
                 ('classifier', LogisticRegression()),
                 ])

loop_time = 1
for loop_idx in range(loop_time):
    # define nested cv
    outer_scores_single, outer_scores_mean, best_params = nested_cv(data_scale, label, run_idx, 
                                                                    Classifier=pipe, param_grid=param_grid,
                                                                    groupby='group_run', #sess=4,
                                                                    )
    print("Cross-validation scores in single trial: ", outer_scores_single)
    print("Mean cross-validation score in single trial: ", np.array(outer_scores_single).mean())
    
    print("Cross-validation scores in mean pattern: ", outer_scores_mean)
    print("Mean cross-validation score in mean pattern: ", np.array(outer_scores_mean).mean())
    
    print("Best params", best_params)
    info.loc[loop_idx, ['single', 'mean']] = [np.array(outer_scores_single).mean(), 
                                              np.array(outer_scores_mean).mean()]
    print(f'Finish loop {loop_idx}')
    
info.to_csv(f'{out_path}/scaler_{sub_name}.csv', index=False)

