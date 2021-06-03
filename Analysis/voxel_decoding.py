import nibabel as nib
import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
import os, pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#%%
n_roi = 180
n_threh = 80
n_run = 40
n_img = 100
n_class = 30

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'

data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-response.mat')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')

data_raw = sio.loadmat(data_path)['response']
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
label_raw = np.load(label_path)
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

n_threh = 80
df_label = pd.DataFrame(label_raw).value_counts()
label_selection = df_label[df_label >= n_threh].index.tolist()
define_label = [x[0] for x in label_selection]
define_label.remove(22)
define_label.remove(26)
label_filter = [True if x in define_label else False for x in label_raw]

#%% Select roi that in visual area

select_network = [11]
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
print(f'Select {voxel_selected.sum()} voxels')

data = data_raw[:, :voxel_selected.shape[0]]
data = data[:, voxel_selected]
print('Finish data loading')


#%% load data

# preprocessing on each run
data = data.reshape(n_run, n_img, n_roi)
data_scale = np.zeros(data.shape)
for run_idx in range(n_run):
    scaler = StandardScaler()
    data_scale[run_idx] = scaler.fit_transform(data[run_idx])
data_scale = data_scale.reshape(n_run*n_img, n_roi)

data_scale = data_scale[label_filter]
label = label_raw[label_filter]

# grid search based on 








# define params for grid search
param_grid = {#'ridge__alpha': [1, 10, 100, 200, 500, 600, 1000, 1e4, 1e5, 1e6],}
              'logisticregression__C': [0.001, 0.01, 0.1, 1], }
              #'svc__C': [1e-6,1e-5,1e-4,1e-3,1e-3,0.01],
              #'svc__gamma': [1e-6,1e-5,1e-4,1e-3,1e-3,0.01],}
              #'pca__n_components':[5, 10, 50, 100]} #

# make pipeline
pipe = make_pipeline(StandardScaler(),
                     #PCA(),
                     LogisticRegression(max_iter=5000),
                     )

# define grid search
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2)
grid.fit(X_train, y_train)

# model = SVC(C=0.01, gamma=0.01)
# model.fit(X_train, y_train)
# results = pd.DataFrame(grid.cv_results_)
# print(results.T)
print('Best cv Score: {:.2f}'.format(grid.best_score_))
print('Test Score: {:.2f}'.format(grid.score(X_test, y_test)))
print('Best parameters: ', grid.best_params_)

y_pred = grid.predict(X_test)





