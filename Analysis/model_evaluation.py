import numpy as np
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
import random
from data_pre import find_outlier

from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import matplotlib.font_manager


#%% define functions
def top_k_acc(X_probs, y_test, class_names, k):
    # top k
    best_n = np.argsort(X_probs, axis=1)[:, -k:]
    y_top_k = class_names[best_n]
    acc_top_k = np.mean(np.array([1 if y_test[n] in y_top_k[n] else 0 for n in range(y_test.shape[0])]))
    return  acc_top_k

#%% define params
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
            
select_network = [1,2,10,11]
scaler = StandardScaler()
        
#% Select roi that in visual area
roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
data = data_raw[:, voxel_selected]
print(f'Select {voxel_selected.sum()} voxels')
label = label_raw[:, 0]
run_idx = label_raw[:, 1]

class_names = np.unique(label)
n_class = class_names.shape[0]
#% =======Enhance data====================
# ======== remove outlier=================
out_index = find_outlier(data, label, 0.05)
data = np.delete(data, out_index, axis=0)
label = np.delete(label, out_index, axis=0)
run_idx = np.delete(run_idx, out_index, axis=0)

# ========preprocessing on each run======
data_scale = np.zeros(data.shape)
for idx in range(n_run):
    tmp_data = data[run_idx==idx, :]
    data_scale[run_idx==idx, :] = scaler.fit_transform(tmp_data)
        
# =======split train and test ==========
# loop_times = 1000

# info = pd.DataFrame(columns=['train_idx', 'acc_mean', 'acc_single'])
# info = info.astype('object')

# for loop_idx in range(loop_times):
train_idx = [32, 38, 18, 4, 3, 21, 8, 9, 10, 28, 17, 29, 27, 2, 20, 13, 
             33, 6, 19, 36, 22, 26, 7, 0, 12, 34, 14, 25, 37, 23]
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
X_test_mean = np.delete(X_test_mean, 0, axis=0)
y_test_mean = np.array(y_test_mean)
# =============== start grid searching==============

# make pipeline
pipe = Pipeline([('feature_selection', SelectPercentile(percentile=50)),
                 ('classifier', LogisticRegression(C=0.001, max_iter=8000)),
                 ])
pipe.fit(X_train, y_train)

# test score
X_probs_mean = pipe.predict_proba(X_test_mean)
X_probs_single = pipe.predict_proba(X_test_single)


print('Test Single Score in Top1: {:.3f}'.format(top_k_acc(X_probs_single, y_test_single, class_names, k=1)))
print('Test Single Score in Top3: {:.3f}'.format(top_k_acc(X_probs_single, y_test_single, class_names, k=3)))
print('Test Single Score in Top5: {:.3f}'.format(top_k_acc(X_probs_single, y_test_single, class_names, k=5)))

print('Test Mean Score in Top1: {:.3f}'.format(top_k_acc(X_probs_mean, y_test_mean, class_names, k=1)))
print('Test Mean Score in Top3: {:.3f}'.format(top_k_acc(X_probs_mean, y_test_mean, class_names, k=3)))
print('Test Mean Score in Top5: {:.3f}'.format(top_k_acc(X_probs_mean, y_test_mean, class_names, k=5)))


#     info.loc[loop_idx, ['train_idx', 'acc_mean', 'acc_single']] = \
#         train_idx, test_mean_score, test_single_score

#     print(f'Finish computing {loop_idx+1} epochs')

# info.to_csv(f'{out_path}/acc_train_test_split_{sub_name}.csv', index=False)


#%% get confusion matrix 
y_pred_single = pipe.predict(X_test_single)
y_pred_mean = pipe.predict(X_test_mean)

single_con = confusion_matrix(y_test_single, y_pred_single, normalize='true')
mean_con = confusion_matrix(y_test_mean, y_pred_mean, normalize='true')


#%% plot 


cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(single_con, cmap=cmap, norm=norm)
plt.colorbar()

plt.xlabel('Predict label', font)
plt.ylabel('True label', font)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test_single),
            fontproperties='arial', weight='bold', size=10)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test_single),
            fontproperties='arial', weight='bold', size=10)
plt.title('Confusion matrix for single trial', font)
plt.savefig(pjoin(out_path, 'confusion_single.jpg'))
plt.close()


#%% 

cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(mean_con, cmap=cmap, norm=norm)
plt.colorbar()

plt.xlabel('Predict label', font)
plt.ylabel('True label', font)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test_single),
            fontproperties='arial', weight='bold', size=10)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test_single),
            fontproperties='arial', weight='bold', size=10)
plt.title('Confusion matrix for mean pattern', font)
plt.savefig(pjoin(out_path, 'confusion_mean.jpg'))
plt.close()


#%% classification report

single_report = classification_report(y_test_single, y_pred_single, output_dict=True)
mean_report = classification_report(y_test_mean, y_pred_mean, output_dict=True)

df_single = pd.DataFrame(single_report).transpose()
df_mean = pd.DataFrame(mean_report).transpose()

df_single.to_csv(pjoin(out_path, 'classification_report_single.csv'))
df_mean.to_csv(pjoin(out_path, 'classification_report_mean.csv'))


#%%

# def plot_confusion_matrix(y_test, y_pred, specify):
#     """

#     Parameters
#     ----------
#     y_test : ndarray
#         Groundtruth class 
#     y_pred : ndarray
#         Class predicted by model
#     specify : str
#         Name to specify this plot

#     """
    
    
#     # get confusion
#     confusion = confusion_matrix(y_test, y_pred, normalize='true')
#     n_class = confusion.shape[0]
#     # visualize
#     cmap = plt.cm.jet
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
#     font = {'family': 'serif', 'weight': 'bold', 'size':14}

#     plt.imshow(confusion, cmap=cmap, norm=norm)
#     plt.colorbar()
    
#     plt.xlabel('Predict label', font)
#     plt.ylabel('True label', font)
#     plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test),
#                fontproperties='arial', weight='bold', size=10)
#     plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), np.unique(y_test),
#                fontproperties='arial', weight='bold', size=10)
#     plt.title(f'Confusion matrix {specify}', font)
#     plt.savefig(pjoin(out_path, f'confusion_{specify}.jpg'))
#     plt.close()
    
    


        
