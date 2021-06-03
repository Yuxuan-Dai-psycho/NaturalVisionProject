import numpy as np
import pandas as pd
from os.path import join as pjoin
import os, pickle
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC

#%% load data and label
# main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
# sub_name = 'sub-core02'

# data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet_roi-response.pkl')
# label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')

# with open(data_path ,'rb') as f:
#     data = pickle.load(f)
# label_raw = np.load(label_path)

# define_label = [19,24,16,6,25,1,20,21,8,26]
# label_filter = [True if x in define_label else False for x in label_raw]
# label = label_raw[label_filter]

# #%% getting decoding results for each roi

# acc_info = pd.DataFrame(columns=['roi', 'acc'])
# # rois = ['V8', 'V4']

# for idx,roi_name in enumerate(data.keys()):
#     data_roi = data[roi_name][label_filter]
#     # define params for grid search
#     param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1], }
#     # make pipeline
#     pipe = make_pipeline(StandardScaler(),
#                          LogisticRegression(max_iter=5000),
#                          )    
#     # define grid search
#     grid = GridSearchCV(pipe, param_grid, cv=5)
#     # get nested cv score
#     scores = cross_val_score(grid, data_roi, label, cv=5)
#     # store info
#     acc_info.loc[idx, ['roi', 'acc']] = [roi_name, scores.mean()]
#     print(f'Finish getting results in ROI: {roi_name}; Acc: {scores.mean()}')

# acc_info.to_csv(pjoin(main_path, 'roi_acc.csv'), index=False)

#%% decoding based on mean pattern on roi

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'
n_roi = 180
n_threh = 80
n_run = 40
n_img = 100
n_class = 30

data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_roi-mean-pattern.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')
animate_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_animate-label.npy')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

roi_raw = pd.read_csv(roi_name_path)
roi_names = [x.split('_')[1] for x in roi_raw.iloc[:, 0].tolist()]
#==========preoare data and label ===============
# load raw label and data
data_raw = np.load(data_path)
label_raw = np.load(label_path)
animate_label_raw = np.load(animate_path)
animate_label = [x[0]+1 if x == [-1] else x[0] for x in animate_label_raw.tolist()]

# select label that its sum samples are larger than 80
df_label = pd.DataFrame(label_raw).value_counts()
label_selection = df_label[df_label >= n_threh].index.tolist()
define_label = [x[0] for x in label_selection]
define_label.remove(22)
define_label.remove(26)

label_filter = [True if x in define_label else False for x in label_raw]

# average left and right brain info
data = np.zeros((data_raw.shape[0], n_roi))
for roi_idx in range(n_roi):
    # get data
    data[:, roi_idx] = np.mean(data_raw[:, [roi_idx, roi_idx+180]], axis=1)

print('Finish data loading')

#%%==========preprocessing on data===============

data = data.reshape(n_run, n_img, n_roi)
data_scale = np.zeros(data.shape)
for run_idx in range(n_run):
    scaler = StandardScaler()
    data_scale[run_idx] = scaler.fit_transform(data[run_idx])
data_scale = data_scale.reshape(n_run*n_img, n_roi)

data_scale = data_scale[label_filter]
label = label_raw[label_filter]

#========== construct model ===============
X_train, X_test, y_train, y_test = train_test_split(data_scale, label, random_state=0)
# define params for grid search
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1], 
              # 'svc__C': [1e-9,1e-8,1e-7],
              # 'svc__gamma': [1e-6],
              }
# make pipeline
pipe = make_pipeline(#StandardScaler(),
                     LogisticRegression(max_iter=8000),
                     #SVC(kernel='rbf'),
                     )    
# define grid search
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

# model = LogisticRegression(C=0.01, max_iter=8000)
# groups = np.repeat(np.linspace(1,40,40,dtype=int), n_img)
# groups = groups[label_filter]
# scores = cross_val_score(model, data_scale, animate_label, groups, cv=GroupKFold(n_splits=5))
# print(scores.mean())

# get test info
print('Best cv Score: {:.3f}'.format(grid.best_score_))
print('Test Score: {:3f}'.format(grid.score(X_test, y_test)))
print('Best parameters: ', grid.best_params_)
y_pred = grid.predict(X_test)

#%% testing based on mean pattern

data = data.reshape(n_run, n_img, n_roi)
data_scale = np.zeros(data.shape)
for run_idx in range(n_run):
    scaler = StandardScaler()
    data_scale[run_idx] = scaler.fit_transform(data[run_idx])
# data_scale = data_scale.reshape(n_run*n_img, n_roi)
# data_scale = data_scale[label_filter]
label = label_raw.reshape(n_run, n_img)
label_filter = np.array(label_filter).reshape(n_run, n_img)

# new_roi_index = [6,17,20,152,158]
# data_scale = data_scale[:,:, new_roi_index]
# n_roi = len(new_roi_index)

n_trains = np.linspace(2,8,7,dtype=int)
test_single_score = np.zeros(n_trains.shape)
test_mean_score = np.zeros(n_trains.shape)
train_score = np.zeros(n_trains.shape)

for idx,n_train in enumerate(n_trains):
    train_runs =[]
    test_runs = []
    for i in range(4):
        train_runs.extend(np.linspace(0,n_train-1,n_train,dtype=int)+10*i)
        test_runs.extend(np.linspace(n_train,9,10-n_train,dtype=int)+10*i)
    
    X_train = data_scale[train_runs].reshape(len(train_runs)*n_img, n_roi)
    y_train = label[train_runs].reshape(len(train_runs)*n_img)
    train_filter = label_filter[train_runs].reshape(len(train_runs)*n_img)
    X_train = X_train[train_filter]
    y_train = y_train[train_filter]
    
    # test on single trial
    X_test_single = data_scale[test_runs].reshape(len(test_runs)*n_img, n_roi)
    y_test_single = label[test_runs].reshape(len(test_runs)*n_img)
    test_filter = label_filter[test_runs].reshape(len(test_runs)*n_img)
    X_test_single = X_test_single[test_filter]
    y_test_single = y_test_single[test_filter]
    
    # test on mean pattern
    X_test_raw = data_scale[test_runs]
    y_test_raw = label[test_runs]
    
    X_test_mean = np.zeros((1, n_roi))
    y_test_mean = []
    
    for run_idx in range(len(test_runs)):
        tmp_X = X_test_raw[run_idx]
        tmp_y = y_test_raw[run_idx]
        for class_idx in range(n_class):
            if (class_idx+1 in define_label) & (class_idx+1 in tmp_y):
                class_loc = tmp_y == class_idx+1
                pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                y_test_mean.append(class_idx)
    X_test_mean= np.delete(X_test_mean, 0, axis=0)
    y_test_mean = np.array(y_test_mean)+1
    
    #========== construct model ===============
    # define model
    model = LogisticRegression(C=1, max_iter=8000)
    # define params for grid search
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 
    #               # 'svc__C': [1e-9,1e-8,1e-7],
    #               # 'svc__gamma': [1e-6],
    #               }
    # define grid search
    # grid = GridSearchCV(model, param_grid, cv=5)
    # grid.fit(X_train, y_train)
    model.fit(X_train, y_train)
    
    # groups = np.repeat(np.linspace(1,40,40,dtype=int), n_img)
    # groups = groups[label_filter]
    # scores = cross_val_score(model, data_scale, animate_label, groups, cv=GroupKFold(n_splits=5))
    # print(scores.mean())
    
    # get test info
    # print('Best cv Score: {:.3f}'.format(grid.best_score_))
    # print('Test Score: {:3f}'.format(grid.score(X_test, y_test)))
    # print('Best parameters: ', grid.best_params_)
    
    # print('Train Score: {:3f}'.format(model.score(X_train, y_train)))
    # print('Single trial Test Score: {:3f}'.format(model.score(X_test_single, y_test_single)))
    # print('Mean pattern Test Score: {:3f}'.format(model.score(X_test_mean, y_test_mean)))
    # y_pred_single = model.predict(X_test_single)
    # y_pred_mean = model.predict(X_test_mean)
    # y_pred_train = model.predict(X_train)
    test_single_score[idx] = model.score(X_test_single, y_test_single)
    test_mean_score[idx] = model.score(X_test_mean, y_test_mean)
    train_score[idx] = model.score(X_train, y_train)
    


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager
font = {'family': 'serif', 'weight': 'bold', 'size':16}
font_legend = {'family': 'serif', 'weight': 'bold', 'size':12}
plt.figure(figsize=(10,6))

plt.plot(train_score)
plt.plot(test_single_score)
plt.plot(test_mean_score)

ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.legend(labels=['train', 'test_single', 'test_mean'], loc='best', prop=font_legend)
plt.xlabel("Total run number of train set", font, labelpad=8)
plt.ylabel("Accuracy", font)
plt.xticks(range(n_trains.shape[0]), n_trains,fontproperties='arial', weight='bold', size=11)
plt.yticks(fontproperties='arial', weight='bold', size=11)
plt.title('Decoding acc of different trainset size', font)
plt.savefig(pjoin(out_path, 'decoding_acc_vva.jpg'))
plt.close()



#%% show model params info
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

model_params = grid.best_estimator_.named_steps['logisticregression'].coef_
roi_max_params = np.max(model_params, axis=0)

for n_select in np.linspace(1,15,15,dtype=int):
    
    select_roi = np.asarray(roi_names)[np.argsort(roi_max_params)[-n_select:]]
    select_pattern = data_scale[:, np.argsort(roi_max_params)[-n_select:]]
    
    X_train, X_test, y_train, y_test = train_test_split(select_pattern, animate_label, random_state=0)
    # define grid search
    grid_new = GridSearchCV(pipe, param_grid, cv=5)
    grid_new.fit(X_train, y_train)
    
    # get test info
    print(f'Select in {n_select} ROIs')
    print('Best cv Score: {:.2f}'.format(grid_new.best_score_))
    print('Test Score: {:.2f}\n'.format(grid_new.score(X_test, y_test)))

