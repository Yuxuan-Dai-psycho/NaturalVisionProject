import nibabel as nib
import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
import os, pickle

#%% load data and label
path = '/nfs/e2/workingshop/swap/imagenet_decoding'
sub_name = 'sub-core02'

data_path = pjoin(path, f'{sub_name}_imagenet-response.mat')
label_path = pjoin(path, f'{sub_name}_imagenet-label.npy')

data_raw = sio.loadmat(data_path)['response']
label_raw = np.load(label_path)
# pd.DataFrame(label_raw).value_counts()

define_label = [19,24,16,6,25,1,20,21,8,26]
label_filter = [True if x in define_label else False for x in label_raw]

label = label_raw[label_filter]
data = data_raw[label_filter]

#%% define roi
import nibabel as nib

df_ROIs = pd.read_csv(pjoin(path, 'VVA_ROIs.csv'))
atlas = sio.loadmat(pjoin(path, 'glasser_atlas.mat'))['glasser2016'].squeeze()

roi_index = df_ROIs['label'].tolist()
voxel_selected = np.asarray([True if x in roi_index else False for x in atlas])

data = data[:, voxel_selected]
print('Finish data loading')


#%% make model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load data
X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)
print('Finish data spliting')

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


