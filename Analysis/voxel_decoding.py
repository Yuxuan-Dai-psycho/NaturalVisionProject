import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio

from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# personal function
from model_utils import nested_cv, find_outlier, gen_param_grid, class_sample

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_id = 2
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)

roi_all = ['V1','V2','V3','V4','V8',
           'FFC','VVC','VMV1','VMV2','VMV3']
for roi_name in roi_all:

    data_path = pjoin(main_path, 'imagenet_decoding', 'roi', f'{sub_name}_imagenet-response_{roi_name}.npy')
    label_path = pjoin(main_path, 'imagenet_decoding','roi', f'{sub_name}_imagenet-label-num_prior.npy')
    out_path = pjoin(main_path, 'imagenet_decoding', 'results')
    
    # data, label and run_idx
    data = np.load(data_path)
    label_raw = np.load(label_path)
    label = label_raw[:, 0]
    run_idx = label_raw[:, 1]
    
    data, label, run_idx = class_sample(data, label, run_idx)
    
    #%% grid search loop
    info = pd.DataFrame(columns=['single', 'mean'])
            
    #% =======Enhance data====================
    # ======== remove outlier=================
    # out_index = find_outlier(data, label, 0.05)
    # data = np.delete(data, out_index, axis=0)
    # label = np.delete(label, out_index, axis=0)
    # run_idx = np.delete(run_idx, out_index, axis=0)
    
    # define dual or primal formulation
    # dual = False if X_train.shape[0] > X_train.shape[1] else True
    # =============== start grid searching==============
    # define params for grid search
    
    
    param_grid = gen_param_grid('logistic')                 
    
    # make pipeline
    pipe = Pipeline([('feature_selection', SelectPercentile(percentile=25)),
                     ('classifier', LogisticRegression(C=0.001, max_iter=8000, solver='liblinear')),
                     ])
    
    ### best params after grid searching ###
    # LogisticRegression(C=0.001, max_iter=8000, solver='liblinear')
    # MLPClassifier(hidden_layer_sizes=100, alpha=0.01)
    # SVC(max_iter=8000, C=0.001, kernel='linear')
    # RandomForestClassifier(n_estimators=500)
    # Lasso(alpha=0.01)
    # LogisticRegression(C=0.001, max_iter=8000, penalty='l1', solver='liblinear')
    

    loop_time = 10
    for loop_idx in range(loop_time):
        # sample data
    
        # define nested cv
        outer_scores_single, outer_scores_mean, best_params = nested_cv(data, label, run_idx, param_grid=param_grid,
                                                                        Classifier=pipe, grid_search=True, k=1,  
                                                                        groupby='group_run', postprocess=False, #sess=4, 
                                                                        )
        print("Cross-validation scores in single trial: ", outer_scores_single)
        print("Mean cross-validation score in single trial: ", np.array(outer_scores_single).mean())
        
        print("Cross-validation scores in mean pattern: ", outer_scores_mean)
        print("Mean cross-validation score in mean pattern: ", np.array(outer_scores_mean).mean())
        
        print("Best params", best_params)
        info.loc[loop_idx, ['single', 'mean']] = [np.array(outer_scores_single).mean(), 
                                                  np.array(outer_scores_mean).mean()]
        print(f'Finish loop {loop_idx} in {roi_name}')
        
    info.loc[loop_idx+1, ['single', 'mean']] = [info.iloc[:,0].mean(), info.iloc[:,1].mean()]
    info.loc[loop_idx+2, ['single', 'mean']] = [info.iloc[:,0].std(), info.iloc[:,1].std()]
    info.to_csv(f'{out_path}/acc/{sub_name}-same_sample-{roi_name}.csv', index=False)


