import json
import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler 


#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
network_path = '/nfs/p1/atlases/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
roi_path = pjoin(main_path, 'MMP_mpmLR32k.mat')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding')

n_run = 40

# network and roi
network = sio.loadmat(network_path)['netassignments']
network = [x[0] for x in network]
roi = sio.loadmat(roi_path)['glasser_MMP']
roi_names = pd.read_csv(roi_name_path)

for sub_id in [2]:
    sub_name = 'sub-{:02d}'.format(sub_id)
    sub_core_name = 'sub-core{:02d}'.format(sub_id)
    data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-beta.mat')
    label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_imagenet-label.npy')
    animacy_label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_core_name}_animate-label.npy')
    
    # data, label and run_idx
    data_raw = sio.loadmat(data_path)['response']
    label = np.load(label_path)
    animacy_label = np.load(animacy_label_path).squeeze()
    animacy_label = np.array([x+3 if x == -1 else x for x in animacy_label])
    run_idx = np.repeat(np.linspace(0,n_run-1,n_run, dtype=int), 100)
    
    # define select class and roi
    class_selected = [1, 6, 8, 9, 16, 18, 19, 
                      20, 24, 25]
    class_loc = np.asarray([True if x in class_selected else False for x in label])
    # roi
    select_network = [1, 2, 10, 11]
    roi_index = [idx+1 for idx,x in enumerate(network) if x in select_network]
    voxel_selected = np.asarray([True if x in roi_index else False for x in roi[0]])
    print(f'Select {voxel_selected.sum()} voxels')
    data = data_raw[class_loc, :roi.shape[1]]
    data = data[:, voxel_selected]
    run_idx = run_idx[class_loc]
    del data_raw
    
    # preprocess for scaling
    scaler = StandardScaler()
    data_scale = np.zeros(data.shape)
    for idx in range(n_run):
        tmp_data = data[run_idx==idx, :]
        # scale on column, based on feature
        tmp_data = scaler.fit_transform(tmp_data)
        # fill data
        data_scale[run_idx==idx, :] = tmp_data
    
    # label and run_idx
    label_all = np.zeros((data.shape[0], 3))
    label_all[:, 0] = label[class_loc]
    label_all[:, 1] = run_idx
    label_all[:, 2] = animacy_label[class_loc]

    # save data
    np.save(pjoin(main_path, 'imagenet_decoding', 'ica', f'{sub_name}_imagenet-beta.npy'), data_scale)
    np.save(pjoin(main_path, 'imagenet_decoding', 'ica', f'{sub_name}_imagenet-label-num_prior.npy'), label_all)

#%% select class based on distance

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-02'

data_path = pjoin(main_path, 'imagenet_decoding', 'class', f'{sub_name}_imagenet-beta-16class.npy')
label_path = pjoin(main_path, 'imagenet_decoding', 'class', f'{sub_name}_imagenet-label-16class.npy')

# load data, label, run_idx
data = np.load(data_path) 
label_all = np.load(label_path)
label = label_all[:, 0]
run_idx = label_all[:, 0]
del label_all

# construct mean pattern
n_class = np.unique(label).shape[0]
n_compare = pd.DataFrame(label).value_counts().min()
n_loop = 1000

class_pattern = np.zeros((n_loop, n_class, data.shape[1]))
# we will do random selection here
# to minimize the randomness, loop 1000 times here
for loop_idx in range(n_loop):
    for idx,class_idx in enumerate(np.unique(label)):
        class_loc = label == class_idx 
        # random select sample to make each class has the same number
        data_sample = data[class_loc, :][np.random.choice(np.sum(class_loc), n_compare, replace=False)]
        class_pattern[loop_idx, idx, :] = np.mean(data_sample, axis=0)
    print('Finish %04d/%04d random selection'%(loop_idx+1, n_loop))
class_pattern = np.mean(class_pattern, axis=0)
del data

np.save(pjoin(main_path, 'imagenet_decoding', 'class', f'{sub_name}_imagenet-16class_pattern.npy'), class_pattern)


#%% construct distance matrix

import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import pairwise_distances

sub_name = 'sub-03'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-class_pattern.npy'))

# correlation distance
corr_matrix = pairwise_distances(class_pattern, metric='correlation')
# corr_matrix[range(n_class), range(n_class)] = 0

plt.figure(figsize=(10,8))
cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(corr_matrix, cmap=cmap)
plt.colorbar()

plt.xlabel('Class idx', font)
plt.ylabel('Class idx', font)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=int), 
           np.unique(label).astype(int),
           fontproperties='arial', weight='bold', size=10)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=int), 
           np.unique(label).astype(int),
           fontproperties='arial', weight='bold', size=10)
plt.title('Distance measured by correlation', font)
plt.savefig(pjoin(out_path, 'corr_RDM.jpg'))
    
corr_distance = np.mean(corr_matrix, axis=0)
order = np.unique(label).astype(int)[np.argsort(-corr_distance)]
print('Distant Order by Correlation (from big to small):\n', order)

#%% Euclidean distance
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.spatial import distance_matrix

sub_name = 'sub-03'
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-class_pattern.npy'))

eucl_matrix =  distance_matrix(class_pattern, class_pattern)

plt.figure(figsize=(10,8))
cmap = plt.cm.jet
font = {'family': 'serif', 'weight': 'bold', 'size':14}
# task cn
plt.imshow(eucl_matrix, cmap=cmap)
plt.colorbar()

plt.xlabel('Class idx', font)
plt.ylabel('Class idx', font)
plt.xticks(np.linspace(0, n_class-1, n_class, dtype=int), 
           np.unique(label).astype(int),
           fontproperties='arial', weight='bold', size=10)
plt.yticks(np.linspace(0, n_class-1, n_class, dtype=int), 
           np.unique(label).astype(int),
           fontproperties='arial', weight='bold', size=10)
plt.title('Distance measured by Euclidean', font)
plt.savefig(pjoin(out_path, 'eucl_RDM.jpg'))
    
eucl_distance = np.mean(eucl_matrix, axis=0)
order = np.unique(label).astype(int)[np.argsort(-eucl_distance)]
print('Distant Order by Euclidean (from big to small):\n', order)

#%% hierarchy clustering
from scipy.cluster import hierarchy 
import matplotlib.font_manager

sub_name = 'sub-03'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-class_pattern.npy'))

# prepare superclass name
with open(pjoin(main_path, 'superclassinfo.json'), 'r') as f:
    super_class_info = json.load(f)
super_class_name = ["%d.%s" % (_[0], _[1]) for _ in super_class_info['info']]
labels_hierarchy = np.array(super_class_name)[np.unique(label).astype(int)-1]

color_list = [f'C{i}' for i in range(10)]
hierarchy.set_link_color_palette(color_list)

method = 'complete'
# fit model
clusters = hierarchy.linkage(class_pattern, method)

# plot 
plt.figure(figsize=(14,10))
info = hierarchy.dendrogram(clusters, labels=labels_hierarchy, orientation='top',
                            leaf_font_size=9, leaf_rotation=60)

font = {'family': 'serif', 'weight': 'bold', 'size':16}
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.xticks(fontproperties='arial', weight='bold', size=11)
plt.yticks(fontproperties='arial', weight='bold', size=11)

plt.xlabel('Class name', font)
plt.ylabel('Distance', font)
plt.title('Hierarchy clustering', font)

plt.savefig(pjoin(out_path, 'hierarchy_cluster.jpg'), bbox_inches='tight')


#%% summarize info on classfication report

classification_path = '/nfs/m1/BrainImageNet/Analysis_results/imagenet_decoding/results/classification_report'
n_split = 10
n_class = 16
score_summary = np.zeros((n_class, 3))

for method_idx,method in enumerate(['mean', 'single']):
    f_score = np.zeros((n_split, n_class))
    for idx in range(n_split):
        file = f'classification_report_{method}_split{idx+1}.csv'
        df = pd.read_csv(pjoin(classification_path, file))
        f_score[idx] = np.array(df.iloc[:n_class, 3])
    score_summary[:, method_idx+1] = np.mean(f_score, axis=0)

score_summary[:, 0] = np.array(df.iloc[:n_class, 0]) 
df_summary = pd.DataFrame(score_summary, columns=('class', 'mean', 'single'))
df_summary.to_csv(pjoin(classification_path, 'score_summary.csv'), index=False)

#%% classfication report and distance bar
# class sample, classification f1 score for all sample, Euclidean distance
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager

sub_name = 'sub-03'
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_selected_org = [1, 6, 8, 9, 12, 14, 16, 17, 
                      18, 19, 20, 21, 22, 24, 25, 26]
n_class = len(class_selected_org)

# get classification f1 score for all sample
classification_path = pjoin(main_path, 'imagenet_decoding', 'results', 'classification_report')
df_summary = pd.read_csv(pjoin(classification_path, 'score_summary.csv'))
score_all = np.array(df_summary.iloc[:n_class, 1]) 

df_summary_same_sample = pd.read_csv(pjoin(classification_path, 'score_summary_same_sample.csv'))
score_sample = np.array(df_summary_same_sample.iloc[:n_class, 1]) 

# get class sample
class_mapping = pd.read_csv(pjoin(main_path, 'superClassMapping.csv'))
super_class_name, super_class_number = [], []
for i in range(30):
    superclass_df = class_mapping.loc[class_mapping['superClassID']==i+1, 'superClassName']
    super_class_name.append("%d.%s" % (i+1, superclass_df.values[0]))
    super_class_number.append(len(superclass_df))
    
labels_selected = np.array(super_class_name)[np.array(class_selected_org)-1]
class_number_selected = np.array(super_class_number)[np.array(class_selected_org)-1]
class_number_norm = class_number_selected/class_number_selected.sum()

# get Euclidean distance
class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-class_pattern.npy'))
eucl_matrix =  distance_matrix(class_pattern, class_pattern)
eucl_distance = np.mean(eucl_matrix, axis=0)
eucl_distance_norm = (eucl_distance - eucl_distance.min()) / (eucl_distance.max() - eucl_distance.min())

# compute corr
r_all_score_num = pearsonr(score_all, class_number_norm)
r_all_score_distance = pearsonr(score_all, eucl_distance_norm)
r_sample_score_num = pearsonr(score_sample, class_number_norm)
r_sample_score_distance = pearsonr(score_sample, eucl_distance_norm)

# plot
x_1 = 3*np.arange(n_class)
x_2 = x_1 + 0.8
x_3 = x_1 - 0.8

# for all sample
plt.figure(figsize=(16,10))
plt.bar(x_3, class_number_norm, label='class_number_ratio', color='#F54748')
plt.bar(x_2, eucl_distance_norm, label='euclidean_distance', color='#233E8B')
plt.bar(x_1, score_all, label='f1_score', color='#1EAE98')

font = {'family': 'arial', 'weight': 'bold', 'size':16}
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.xticks((x_1 + x_2 + x_3)/3, labels_selected, rotation=45,
           fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)
plt.legend(prop=font)

plt.savefig(pjoin(out_path, 'decoding_info_all.jpg'), bbox_inches='tight')
plt.close()

# for same sample
plt.figure(figsize=(16,10))
plt.bar(x_3, class_number_norm, label='class_number_ratio', color='#F54748')
plt.bar(x_2, eucl_distance_norm, label='euclidean_distance', color='#233E8B')
plt.bar(x_1, score_sample, label='f1_score', color='#1EAE98')

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.xticks((x_1 + x_2 + x_3)/3, labels_selected, rotation=45,
           fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)
plt.legend(prop=font)

plt.savefig(pjoin(out_path, 'decoding_info_same.jpg'), bbox_inches='tight')
plt.close()


#%% plot bar of distance
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager


main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')
class_selected_org = [1, 6, 8, 9, 12, 14, 16, 17, 
                      18, 19, 20, 21, 22, 24, 25, 26]
n_class = len(class_selected_org)

# get class sample
class_mapping = pd.read_csv(pjoin(main_path, 'superClassMapping.csv'))
super_class_name, super_class_number = [], []
for i in range(30):
    superclass_df = class_mapping.loc[class_mapping['superClassID']==i+1, 'superClassName']
    super_class_name.append("%d.%s" % (i+1, superclass_df.values[0]))
    super_class_number.append(len(superclass_df))
labels_selected = np.array(super_class_name)[np.array(class_selected_org)-1]

eucl_distance = np.zeros((n_class, 2))
for idx,sub_id in enumerate([2, 3]):
    sub_name = 'sub-%02d'%sub_id
    # get Euclidean distance
    class_pattern = np.load(pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-class_pattern.npy'))
    eucl_matrix =  distance_matrix(class_pattern, class_pattern)
    eucl_distance[:, idx] = np.mean(eucl_matrix, axis=0)

x_1 = 2.5*np.arange(n_class)
x_2 = x_1 + 0.8

# for all sample
plt.figure(figsize=(16,10))
plt.bar(x_1, eucl_distance[:, 0], label='sub-02', color='#F54748')
plt.bar(x_2, eucl_distance[:, 1], label='sub-03', color='#0A81AB')

font_title = {'family': 'arial', 'weight': 'bold', 'size':20}
font_other = {'family': 'arial', 'weight': 'bold', 'size':16}
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.xticks((x_1 + x_2)/2, labels_selected, rotation=45,
           fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)
plt.legend(prop=font_other)

plt.ylabel('distance', font_other)
plt.title('Euclidean distance of different class', font_title)
plt.savefig(pjoin(out_path, 'distance_bar.jpg'), bbox_inches='tight')
plt.close()




