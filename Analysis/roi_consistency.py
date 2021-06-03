import numpy as np
import pandas as pd
from os.path import join as pjoin
import os, pickle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

#%% load data and label
main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
sub_name = 'sub-core02'
n_run = 40
n_img = 100
n_roi = 180
n_class = 30

data_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet_roi-mean-response.npy')
label_path = pjoin(main_path, 'imagenet_decoding', f'{sub_name}_imagenet-label.npy')
roi_name_path = pjoin(main_path, 'roilbl_mmp.csv')
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

data = np.load(data_path)
label = np.load(label_path)
roi_raw = pd.read_csv(roi_name_path)
roi_names = [x.split('_')[1] for x in roi_raw.iloc[:, 0].tolist()]


#%% find label that appear in all 40 runs
# Method 1: First reshape to 40 runs
# data = data.reshape(n_run, n_img, n_roi)
# label = label.reshape(n_run, n_img)
# odd_idx = np.linspace(1,39,20, dtype=int)-1
# odd_sum = data[odd_idx].reshape(2000, n_roi)
# odd_label = label[odd_idx, :].flatten()

# even_idx = np.linspace(1,39,20, dtype=int)
# even_sum = data[even_idx].reshape(2000, n_roi)
# even_label = label[even_idx, :].flatten()
# del data, label

# Method 2: Directly extract pattern
odd_idx = []
even_idx = []
for idx in np.linspace(1,39,20, dtype=int):
    odd_idx.extend(np.linspace(100*(idx-1),100*idx-1,100, dtype=int).tolist())
    even_idx.extend(np.linspace(100*idx,100*(idx+1)-1,100, dtype=int).tolist())

odd_sum = data[odd_idx, :]
odd_label = label[odd_idx]
even_sum = data[even_idx]
even_label = label[even_idx]

# initialize pattern for class
odd_pattern = np.zeros((n_roi, n_class))
even_pattern = np.zeros((n_roi, n_class))

for class_idx in range(n_class):
    odd_loc = np.where(odd_label==class_idx+1)[0]
    odd_tmp_pattern = np.mean(odd_sum[odd_loc,:], axis=0)
    odd_pattern[:, class_idx] = odd_tmp_pattern 
    
    even_loc = np.where(even_label==class_idx+1)[0]
    even_tmp_pattern = np.mean(even_sum[even_loc,:], axis=0)
    even_pattern[:, class_idx] = even_tmp_pattern 

#%% plot output

corr_matrix = np.corrcoef(odd_pattern, even_pattern)[n_roi:n_roi*2, 0:n_roi]

cmap = plt.cm.jet
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
font = {'family': 'serif', 'weight': 'bold', 'size':12}
# task cn
# plt.figure(figsize=(20,20))
plt.imshow(corr_matrix, cmap=cmap)#, norm=norm)
plt.colorbar()

plt.xlabel("ROI pattern in odd run", font)
plt.ylabel("ROI pattern in even run", font)
# plt.xticks(np.linspace(0, 179, 180, dtype=int), roi_names, fontproperties='serif', weight='bold', size=6)
# plt.yticks(np.linspace(0, 179, 180, dtype=int), roi_names, fontproperties='serif', weight='bold', size=6)
plt.title('Correlation matrix in odd and even run patterns', font)
plt.savefig(pjoin(out_path, 'corr_odd_even.jpg'))
plt.close()


#%% save snr info

diag = [row[i] for i,row in enumerate(corr_matrix)]
snr_info = pd.DataFrame({'roi': roi_names, 'snr': diag})
snr_info.to_csv(pjoin(out_path, 'snr_info.csv'), index=False)


#%% plot class pattern

roi_pattern = np.zeros((n_roi, n_class))
font = {'family': 'serif', 'weight': 'bold', 'size':16}
plt.figure(figsize=(10,6))

for class_idx in range(n_class):
    loc = np.where(label==class_idx+1)[0]
    class_pattern = np.mean(data[loc,:], axis=0)
    roi_pattern[:, class_idx] = class_pattern
    # plot 
    plt.plot(range(n_roi), class_pattern)

ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# plt.legend(labels=[f'class {x+1}' for x in range(n_class)], loc='best')
plt.xlabel("ROI index", font, labelpad=8)
plt.ylabel("Brain Response", font)
plt.xticks(fontproperties='arial', weight='bold', size=11)
plt.yticks(fontproperties='arial', weight='bold', size=11)
plt.title('Class pattern in different ROIs', font)
plt.savefig(pjoin(out_path, 'class_pattern.jpg'))
plt.close()



#%% select some roi to plot class pattern

roi_diff = np.zeros(n_roi)

for roi_idx in range(n_roi):
    tmp_pattern = roi_pattern[roi_idx]
    diff_index = tmp_pattern.std() #tmp_pattern.max() - tmp_pattern.min()
    roi_diff[roi_idx] = diff_index

n_select = 15
select_roi = np.asarray(roi_names)[np.argsort(roi_diff)[-n_select:]]
select_pattern = roi_pattern[np.argsort(roi_diff)[-n_select:]]
plt.figure(figsize=(10,6))

n_class_new = 26
for class_idx in range(n_class_new):
    tmp_pattern = select_pattern[:, class_idx]
    # plot 
    plt.plot(range(n_select), tmp_pattern)

ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.legend(labels=[f'class {x+1}' for x in range(n_class)], loc='best', ncol=2)
plt.xlabel("ROI", font, labelpad=8)
plt.ylabel("Brain Response", font)
plt.xticks(np.linspace(0,n_select-1,n_select, dtype=int), select_roi, fontproperties='arial', weight='bold', size=11)
plt.yticks(fontproperties='arial', weight='bold', size=11)
plt.title('Class pattern in different ROIs', font)
plt.savefig(pjoin(out_path, 'select_class_pattern.jpg'))
plt.close()
    

