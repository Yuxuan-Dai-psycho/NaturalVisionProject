import os
import torch
import numpy as np
import pandas as pd
from os.path import join as pjoin

from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import random_split
# personal function
from model_utils import nested_cv, gen_param_grid, class_sample, voxel_selection, \
                        Dataset, compute_acc, train, plot_training_curve
from brain_inspired_nn import VisualNet, VisualNet_simple, VisualNet_fully_connected

#%% load data and label
main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/'
# for sub_id in [1,4,5,6,8,9,10]:
#     sub_name = 'sub-{:02d}'.format(sub_id)
#     sub_core_name = 'sub-core{:02d}'.format(sub_id)
sub_name = 'sub-04'
data_path = pjoin(main_path, 'imagenet_decoding', 'group', f'{sub_name}_imagenet-beta.npy')
label_path = pjoin(main_path, 'imagenet_decoding', 'group', f'{sub_name}_imagenet-label-num_prior.npy')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# data, label and run_idx
data = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]
sess_idx = label_raw[:, 2]
sub_idx = label_raw[:, 3]
# animacy_label = label_raw[:, 2]

groups = run_idx
# sample selection and voxel selection
# data, label, groups = class_sample(data, label, groups)
print(f'Select {data.shape[0]} samples')
voxel_selection_method = 'stability'
if voxel_selection_method == 'stability':
    voxel_select_percentage = 25
    n_voxel_select = int(data.shape[1]*(voxel_select_percentage/100))
    stability_idx_path = pjoin(main_path, 'imagenet_decoding', 'voxel', 
                                 f'{sub_name}_imagenet-stability_idx.npy')
    if not os.path.exists(stability_idx_path):
        data, stability_idx = voxel_selection(data, label, groups, percentage=voxel_select_percentage)
        np.save(stability_idx_path, stability_idx)
    else:
        stability_idx = np.load(stability_idx_path)
        data = data[:, stability_idx]

#%% Decoding Main Part

decoding_method = 'nn'
#======================== For Sklearn Classifier =======================
if decoding_method == 'sklearn':
    info = pd.DataFrame(columns=['single', 'mean'])
    param_grid = gen_param_grid('lda')                 
    
    # make pipeline
    if voxel_selection_method == 'stability':
        pipe = Pipeline([('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9))
                         ])
    elif voxel_selection_method == 'discrim':
        pipe = Pipeline([('feature_selection', SelectPercentile(percentile=25)),
                         ('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9))
                         ])
    # model = LogisticRegression(C=0.001, max_iter=8000, solver='liblinear')
    # selector = RFE(model, n_features_to_select=0.25)
    
    ### best params after grid searching ###
    # LogisticRegression(C=0.001, max_iter=8000, solver='liblinear')
    # MLPClassifier(hidden_layer_sizes=100, alpha=0.01)
    # SVC(max_iter=8000, C=0.001, kernel='linear', decision_function_shape='ovo')
    # RandomForestClassifier(n_estimators=500)
    # Lasso(alpha=0.01)
    # LogisticRegression(C=0.001, max_iter=8000, penalty='l1', solver='liblinear')
    # LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
    
    loop_time = 5
    for loop_idx in range(loop_time):
        # sample data
    
        # define nested cv
        outer_scores_single, outer_scores_mean, best_params = nested_cv(data, label, groups, 
                                                                        param_grid=param_grid, Classifier=pipe, 
                                                                        grid_search=False, k=1, mean_times=None,
                                                                        groupby='group_run', postprocess=False, #sess=4,
                                                                        #feature_selection=25,
                                                                        )
        print("Cross-validation scores in single trial: ", outer_scores_single)
        print("Mean cross-validation score in single trial: ", np.array(outer_scores_single).mean())
        
        print("Cross-validation scores in mean pattern: ", outer_scores_mean)
        print("Mean cross-validation score in mean pattern: ", np.array(outer_scores_mean).mean())
        
        print("Best params", best_params)
        info.loc[loop_idx, ['single', 'mean']] = [np.array(outer_scores_single).mean(), 
                                                  np.array(outer_scores_mean).mean()]
        print(f'Finish loop {loop_idx}')
        
    info.loc[loop_idx+1, ['single', 'mean']] = [info.iloc[:,0].mean(), info.iloc[:,1].mean()]
    info.loc[loop_idx+2, ['single', 'mean']] = [info.iloc[:,0].std(), info.iloc[:,1].std()]
    info.to_csv(f'{out_path}/acc/{sub_name}-same_sample-test.csv', index=False)

#========================== For Neural Network ===================
elif decoding_method == 'nn':
    # define params
    lr = 0.001
    n_epoch = 30
    train_percentage = 0.8
    batch_size = 32
    weight_decay = 0
    p = 0.5
    verbose = False # if True the code will show the loss info in each batch
    train_size = int(data.shape[0] * train_percentage)
    val_size = data.shape[0] - train_size
    # define model and make dataset
    # model = VisualNet(p, selected_voxel=stability_idx)
    model = VisualNet_simple(p, n_hidden=20)
    dataset = Dataset(data, label)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    # train model
    model_params, train_acc, train_loss, val_acc, val_loss = \
        train(model, train_set, val_set, batch_size, n_epoch, lr, weight_decay)
    # save and plot info
    plot_training_curve(n_epoch, train_acc, train_loss, val_acc, val_loss, 
                        flag=sub_name+'_simple')
    # torch.save(model_params, pjoin(out_path, 'visual_nn.pkl'))
    