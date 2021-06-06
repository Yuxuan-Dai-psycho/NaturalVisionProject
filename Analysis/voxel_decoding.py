import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
import multiprocessing
import random
import data_pre

from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%% define params
n_class = 30
n_run = 40
n_train = 30
n_test = 10

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'

network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_14class-cortex.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_14class-label-runidx.npy')
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# data, label and run_idx
data_raw = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

#%% grid search loop
if __name__ == '__main__':
    # prevent gridsearch multipreocessing loop to hang
    multiprocessing.set_start_method('forkserver')
        
    info = pd.DataFrame(columns=['network', 'scaler', 'model', 'percentile', 
                                 'acc_cv', 'acc_test_single', 'acc_test_mean'])
    
    network_names = {'primary':[1], 'secondary':[2], 'post_multi':[10], 
                     'ventral_multi':[11], 'all':[1,2,10,11]}
    scalers = {'standard':StandardScaler() , 'minmax':MinMaxScaler()}
    loop_idx = 0
    loop_all = len(network_names.keys())*len(scalers.keys())
    for net in network_names.keys():
        select_network = network_names[net]
        for scaler_name in scalers.keys():
            scaler = scalers[scaler_name]
            #% Select roi that in visual area
            roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
            voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
            data = data_raw[:, voxel_selected]
            print(f'Select {voxel_selected.sum()} voxels')
            
            #% =======Enhance data====================
            # ========preprocessing on each run======
            data_scale = np.zeros(data.shape)
            for idx in range(n_run):
                tmp_data = data[run_idx==idx, :]
                data_scale[run_idx==idx, :] = scaler.fit_transform(tmp_data)
                    
            # =======split train and test ==========
            train_idx = random.sample(range(n_run), n_train)
            test_idx = list(set(range(n_run)).difference(set(train_idx)))
            # prepare train set
            train_filter = np.array([True if x in train_idx else False for x in run_idx])
            X_train = data_scale[train_filter]
            y_train = label[train_filter]
            # prepare test set
            # single trial test
            test_filter = np.array([True if x in test_idx else False for x in run_idx])
            X_test_single = data_scale[test_filter]
            y_test_single = label[test_filter]
            # mean pattern test
            X_test_mean = np.zeros((1, data_scale.shape[1]))
            y_test_mean = []
            for idx in test_idx:
                tmp_X = data_scale[run_idx==idx, :]
                tmp_y = label[run_idx==idx]
                for class_idx in np.unique(label):
                    if class_idx in tmp_y:
                        class_loc = tmp_y == class_idx
                        pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                        X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                        y_test_mean.append(class_idx)
            X_test_mean= np.delete(X_test_mean, 0, axis=0)
            y_test_mean = np.array(y_test_mean)
            print('Finish data preparation')
            # define dual or primal formulation
            # dual = False if X_train.shape[0] > X_train.shape[1] else True
            # =============== start grid searching==============
            # define params for grid search
            param_grid = [
                          {'classifier': [LinearSVC(max_iter=8000)], 'feature_selection':[SelectPercentile()],
                           'classifier__C': [0.001, 0.01, 0.1, 1],
                           'feature_selection__percentile': [5,10,20,50,100],},
                          {'classifier': [LogisticRegression(max_iter=8000)], 
                           'feature_selection':[SelectPercentile()],
                           'classifier__C': [0.001, 0.01, 0.1, 1],
                           'feature_selection__percentile': [5,10,20,50,100],},]                    
            
            # make pipeline
            pipe = Pipeline([('feature_selection', SelectPercentile()),
                             ('classifier', LogisticRegression(max_iter=8000)),
                             ])
            
            # define grid search
            grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=8, verbose=10)
            grid.fit(X_train, y_train)
            
            # test score
            cv_score = grid.best_score_
            test_single_score = grid.score(X_test_single, y_test_single)
            test_mean_score = grid.score(X_test_mean, y_test_mean)
            best_params = grid.best_params_
            # save info
            info.loc[loop_idx, ['network', 'scaler', 'model', 'percentile', 
                                'acc_cv', 'acc_test_single', 'acc_test_mean']] = \
                net, scaler_name, best_params['classifier'], best_params['feature_selection'].percentile, \
                cv_score, test_single_score, test_mean_score
            print(f'Finish Grid in {loop_idx+1}/{loop_all}')
            loop_idx += 1
    
            print('Best cv Score: {:.2f}'.format(cv_score))
            print('Test Single Score: {:.2f}'.format(test_single_score))
            print('Test Mean Score: {:.2f}'.format(test_mean_score))
            print('Best parameters: ', grid.best_params_)
            
            
            # y_pred_single = grid.predict(X_test_single)
            # y_pred_mean = grid.predict(X_test_mean)
            # y_pred_train = grid.predict(X_train)
    
    info.to_csv(f'{out_path}/grid_search.csv', index=False)

