import os
import time
import torch
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

#%% Define params
# define path
dataset_path = '/nfs/e3/VideoDatabase/HACS/training'
working_path = '/nfs/s2/userhome/zhouming/workingdir/Video/HACS/stimulus_select'
act_path = '/nfs/s2/userhome/zhouming/workingdir/Video/HACS/train_model/out'
fc_path = f'{act_path}/fc'
input_path = f'{act_path}/input'

#%% Define functions
def util_plot_hist(arr, xlabel, ylabel, title, save_path=None):
    """
    Plot utils function by Blink621

    Parameters
    ----------
    arr : np.ndarray
    xlabel : str
    ylabel : str
    title : str
    save_path : str

    """
    color = '#008891'
    plt.hist(arr, color=color)
    plt.xticks(fontproperties='Arial', size=12)
    plt.yticks(fontproperties='Arial', size=12)
    # prepare axis
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    # label
    font = {'family': 'serif', 'weight': 'normal', 'size':14}
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    plt.title(title, font)
    if save_path != None:
        plt.savefig(save_path)
    plt.close()

    
#%% Compute classification accuracy
# load fc label idx
train_set = pd.read_csv(f'{working_path}/HACS_clips_v1.1_train.csv')
class_label = train_set['label'].unique()
del train_set

acc_info = pd.DataFrame(columns=['class_name', 'class_label', 'video_num', 
                                 'acc1', 'acc5', 'all_positive'])

for idx,act_name in enumerate(os.listdir(fc_path)):
    # define path
    class_name = act_name.replace('_fc_tensor.pt','')
    class_idx = np.where(class_label==class_name)[0][0]
    tensor_path = os.path.join(fc_path, act_name)
    video_path = os.path.join(input_path, f'{class_name}_input_path.npy')
    # load data
    video_act = torch.load(tensor_path)
    video_names = np.load(video_path)
    video_labels = [1 if x.split('_')[-1].split('.')[0]=='1' else 0 for x in video_names] # correspond positive or negative
    neg_or_pos = -1 if 0 in video_labels else 1
    # filter negative clips
    video_act = video_act[torch.Tensor(video_labels)==1, :]
    video_num = video_act.shape[0]
    # get top1 and top5 act in each sample
    _, pred1 = video_act.topk(1,1)
    _, pred5 = video_act.topk(5,1)
    # get acc1 and acc5
    acc1 = round(np.sum(pred1.numpy()==class_idx)/video_num, 2)
    acc5 = round(np.sum(pred5.numpy()==class_idx)/video_num, 2)
    
    acc_info.loc[idx, ['class_name', 'class_label', 'video_num', 'acc1', 'acc5', 'all_positive']] = \
                       [class_name, class_idx, video_num, acc1, acc5, neg_or_pos]
    print(f"Finish checking label for class {class_name}")
    # plot activation distribution
    util_plot_hist(video_act[:,class_idx].numpy(), xlabel='Activation', ylabel='Numbers', 
                   title=f'{class_name}',
                   save_path=f'{working_path}/img_distribution/{class_name}.jpg')
    
    
dataset_num = acc_info['video_num'].sum()

acc_sum1 = ((acc_info['video_num']*acc_info['acc1']).astype('int64')).sum()/dataset_num
acc_sum5 = ((acc_info['video_num']*acc_info['acc5']).astype('int64')).sum()/dataset_num

acc_info.to_csv(f'{working_path}/acc_info.csv', index=False)



#%% Random sample selector
# define params
time_search = 1000
num_exp = 480
num_bins = 40
r_threshold = 0.95

# load fc label idx
train_set = pd.read_csv(f'{working_path}/HACS_clips_v1.1_train.csv')
class_label = train_set['label'].unique()

# choose the class that positive clips are enough
acc_info = pd.read_csv(f'{working_path}/acc_info.csv')
class_positive = acc_info.loc[acc_info['all_positive']==1, 'class_name'].to_list()

# prepare filter info
data = pd.read_csv(f'{working_path}/dataset.csv')
threshold = (data['frame_ratio'].mean() - 3*data['frame_ratio'].std(),
             data['frame_ratio'].mean() + 3*data['frame_ratio'].std())
data_filtered = data.loc[(data['frame_ratio'] > threshold[0]) & \
                         (data['frame_ratio'] < threshold[1]) & \
                         (data['frame_ratio'] == data['crop_ratio'])]

del acc_info, data, train_set

# prepare action dataset dataframe before loading 
action_dataset = pd.DataFrame(columns=['class_name', 'video_name'])

# start random selecting
for idx, class_name in enumerate(class_positive):
    # prepare path
    act_path = os.path.join(fc_path, f'{class_name}_fc_tensor.pt')
    video_path = os.path.join(input_path, f'{class_name}_input_path.npy')
    # load data
    class_idx = np.where(class_label==class_name)[0][0]
    video_act = torch.load(act_path)
    video_act = video_act[:, class_idx].numpy()
    video_names = np.load(video_path)
    # filter data according to length-width ratio
    video_names = [x.split('/')[-1] for x in video_names]
    video_valid = data_filtered.loc[data_filtered['class']==class_name, 'video'].to_list()
    # for those classes that less than 480 classes
    if len(video_valid) < num_exp:
        video_valid = video_names
    else: # class num > 480
        video_valid_flag = [1 if x in video_valid else 0 for x in video_names]
        video_act = video_act[np.asarray(video_valid_flag)==1]
    video_num = video_act.shape[0]
    # compute population hist 
    pop_hist = np.histogram(video_act, bins=num_bins)
    
    time_spent = 0
    time_start = time.time()
    # start random sampling
    while time_spent < time_search:
        # random seed
        np.random.seed()
        sample_list = np.random.choice(video_num, num_exp, replace=False)
        sample_act = video_act[sample_list]
        sample_hist = np.histogram(sample_act, bins=num_bins)
        # compute pearson-r
        act_r = round(stats.pearsonr(pop_hist[0], sample_hist[0])[0], 2)
        if act_r > r_threshold:
            time_spent = time.time() - time_start
            break
    print('Class %03d: pearson-r: %.2f; Spent time: %.2fs in %s'% \
          (idx+1, act_r, time_spent, class_name))
    # store the correspond videos
    # and save them in disk
    video_dataset = np.asarray(video_valid)[sample_list]
    class_dataset = np.repeat([class_name], num_exp)
    tmp_dataset = pd.DataFrame({'class_name':class_dataset, 'video_name':video_dataset})
    action_dataset = action_dataset.append(tmp_dataset, ignore_index=True)
    # copy video into a folder
    fMRI_dataset_class_path = f'{working_path}/action/{class_name}'
    if not os.path.exists(fMRI_dataset_class_path):
        os.makedirs(fMRI_dataset_class_path)
    for video_idx,video_name in enumerate(video_dataset):
        shutil.copy(f'{dataset_path}/{class_name}/{video_name}',
                    f'{fMRI_dataset_class_path}/{video_name}')
    del video_dataset, class_dataset, tmp_dataset
action_dataset.to_csv(f'{working_path}/action_dataset.csv', index=False)


