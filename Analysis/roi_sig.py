import numpy as np
import pandas as pd
from os.path import join as pjoin
import os, pickle
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'
n_run = 40
n_img = 100
n_roi = 180
n_class = 30

data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_roi-mean-pattern.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')

roi_raw = pd.read_csv(roi_name_path)
roi_names = [x.split('_')[1] for x in roi_raw.iloc[:, 0].tolist()]
#==========preoare data and label ===============
# load raw label and data
data_raw = np.load(data_path)
label_raw = np.load(label_path)


# average left and right brain info
data = np.zeros((data_raw.shape[0], n_roi))
for roi_idx in range(n_roi):
    # get data
    data[:, roi_idx] = np.mean(data_raw[:, [roi_idx, roi_idx+180]], axis=1)

data = data.reshape(n_run, n_img, n_roi)
label = label_raw.reshape(n_run, n_img)
del data_raw


#%% Starting anova
run_pattern = np.full((n_run, n_roi, n_class), np.nan)
run_cols = np.repeat(np.linspace(1,40,40,dtype=int),n_roi)
roi_all = np.tile(roi_names ,n_run)

for run_idx in range(n_run):
    org_pattern = data[run_idx]
    run_label = label[run_idx]
    # start merging
    for class_idx in range(n_class):
        if class_idx+1 in run_label:
            label_loc = run_label == class_idx+1
            # plot 
            class_value = np.mean(org_pattern[label_loc, :], axis=0)
            run_pattern[run_idx, :, class_idx] = class_value 
    print(f'Finish merging {run_idx+1} runs')
run_pattern = run_pattern.reshape(n_run*n_roi, n_class)

df_response = pd.DataFrame({'run':run_cols, 'ROI':roi_all})
df_response = pd.concat([df_response, pd.DataFrame(run_pattern)], axis=1)
df_response.columns = ['run', 'ROI', *[f'class{x+1}' for x in range(n_class)]]

