import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
from scipy.spatial import distance_matrix

from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

# personal function
from model_utils import nested_cv, find_outlier, gen_param_grid, class_sample

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_id = 3
sub_name = 'sub-{:02d}'.format(sub_id)
sub_core_name = 'sub-core{:02d}'.format(sub_id)

data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-response_org_preprocess.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label&run_idx_org_preprocess.npy')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

# data, label and run_idx
data = np.load(data_path)
label_raw = np.load(label_path)
label = label_raw[:, 0]
run_idx = label_raw[:, 1]

# sample class
n_sample = pd.DataFrame(label).value_counts().min()
data_sample = np.zeros((1, data.shape[1]))
run_idx_sample = np.zeros((1))
# loop to sample
for idx,class_idx in enumerate(np.unique(label)):
    class_loc = label == class_idx 
    class_data = data[class_loc]
    class_mean = np.mean(class_data, axis=0)[np.newaxis, :]
    eucl_matrix =  distance_matrix(class_data, class_mean)
    eucl_distance = np.mean(eucl_matrix, axis=0)
    # select_idx = np.argsort(eucl_distance)
    # random select sample to make each class has the same number
    select_idx = np.random.choice(np.sum(class_loc), n_sample, replace=False)
    data_class = data[class_loc, :][select_idx]
    run_idx_class = run_idx[class_loc][select_idx]
    # concatenate on the original array
    data_sample = np.concatenate((data_sample, data_class), axis=0)
    run_idx_sample = np.concatenate((run_idx_sample, run_idx_class), axis=0)
# prepare final data
data_sample = np.delete(data_sample, 0, axis=0)
run_idx_sample = np.delete(run_idx_sample, 0, axis=0)
label_sample = np.repeat(np.unique(label), n_sample)
    



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
pipe = Pipeline([('feature_selection', SelectPercentile()),
                 ('classifier', LogisticRegression()),
                 ])

loop_time = 1
for loop_idx in range(loop_time):
    # sample data
    # data_sample, label_sample, run_idx_sample = class_sample(data, label, run_idx)

    # define nested cv
    outer_scores_single, outer_scores_mean, best_params = nested_cv(data, label, run_idx, 
                                                                    Classifier=pipe, param_grid=param_grid,
                                                                    groupby='group_run', postprocess=True, #sess=4, 
                                                                    )
    print("Cross-validation scores in single trial: ", outer_scores_single)
    print("Mean cross-validation score in single trial: ", np.array(outer_scores_single).mean())
    
    print("Cross-validation scores in mean pattern: ", outer_scores_mean)
    print("Mean cross-validation score in mean pattern: ", np.array(outer_scores_mean).mean())
    
    print("Best params", best_params)
    info.loc[loop_idx, ['single', 'mean']] = [np.array(outer_scores_single).mean(), 
                                              np.array(outer_scores_mean).mean()]
    print(f'Finish loop {loop_idx}')
    
info.to_csv(f'{out_path}/sample_{sub_name}.csv', index=False)

