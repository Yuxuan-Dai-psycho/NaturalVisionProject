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

# personal function
from model_utils import nested_cv, gen_param_grid, class_sample, voxel_selection
from brain_inspired_nn import VisualNet

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
# for sub_id in [1,4,5,6,8,9,10]:
#     sub_name = 'sub-{:02d}'.format(sub_id)
#     sub_core_name = 'sub-core{:02d}'.format(sub_id)
sub_name = 'sub-01-10'
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
data, label, groups = class_sample(data, label, groups)
print(f'Select {data.shape[0]} samples')
stability = False
if stability:
    voxel_select_percentage = 25
    n_voxel_select = int(data.shape[1]*(voxel_select_percentage/100))
    stability_score_path = pjoin(main_path, 'imagenet_decoding', 'voxel', 
                                 f'{sub_name}_imagenet-stability_score.npy')
    if not os.path.exists(stability_score_path):
        data, stability_score = voxel_selection(data, label, run_idx, percentage=voxel_select_percentage)
        np.save(stability_score_path, stability_score)
    else:
        stability_score = np.load(stability_score_path)
        stability_idx = np.argsort(stability_score)[:n_voxel_select]
        data = data[:, stability_idx]

#%% Decoding Main Part

method = 'sklearn'
#======================== For Sklearn Classifier =======================
if method == 'sklearn':
    info = pd.DataFrame(columns=['single', 'mean'])
    param_grid = gen_param_grid('lda')                 
    
    # make pipeline
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
                                                                        groupby='group_sub', postprocess=False, #sess=4,
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
    info.to_csv(f'{out_path}/acc/{sub_name}-same_sample-discrim-lda_group_sub.csv', index=False)

#========================== For Neural Network ===================
elif method == 'nn':
    model = VisualNet()
    # define params
    lr = 0.1
    n_epoch = 20
    # create input x and output y
    x = torch.tensor(data)
    y = torch.randn(label)
    # backward pass
    print('======Start training======')
    for t in range(n_epoch):
        # forward
        y_pred = model(x)
        # loss
        loss = (y_pred - y).abs().mean()
        # Zero the gradients before running the backward pass.
        model.zero_grad()
        # Use autograd to compute the backward pass
        loss.backward()
        # Update the weights
        with torch.no_grad():
            for param in model.parameters():
                # mask is also saved in param, but mask.requires_grad=False
                if param.requires_grad: 
                    param -= lr * param.grad

    

