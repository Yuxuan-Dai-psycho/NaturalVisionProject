import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%% Define params
# define path
dataset_path = '/nfs/e3/VideoDatabase/HACS/training'
working_path = '/nfs/s2/userhome/zhouming/workingdir/Video/HACS/stimulus_select'
act_path = '/nfs/s2/userhome/zhouming/workingdir/Video/HACS/train_model/out'
fc_path = f'{act_path}/fc'
input_path = f'{act_path}/input'

def plot_util(arr, xlabel, ylabel, title, save_path=None):
    """
    Plot utils function by Blink621

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    save_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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

    
#%% Check fc label and compute classification accuracy

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
    act_tensor = torch.load(tensor_path)
    video_names = np.load(video_path)
    video_labels = [1 if x.split('_')[-1].split('.')[0]=='1' else 0 for x in video_names] # correspond positive or negative
    neg_or_pos = -1 if 0 in video_labels else 1
    # filter negative clips
    act_tensor = act_tensor[torch.Tensor(video_labels)==1, :]
    video_num = act_tensor.shape[0]
    # get top1 and top5 act in each sample
    _, pred1 = act_tensor.topk(1,1)
    _, pred5 = act_tensor.topk(5,1)
    # get acc1 and acc5
    acc1 = round(np.sum(pred1.numpy()==class_idx)/video_num, 2)
    acc5 = round(np.sum(pred5.numpy()==class_idx)/video_num, 2)
    
    acc_info.loc[idx, ['class_name', 'class_label', 'video_num', 'acc1', 'acc5', 'all_positive']] = \
                       [class_name, class_idx, video_num, acc1, acc5, neg_or_pos]
    print(f"Finish checking label for class {class_name}")
    # plot activation distribution
    plot_util(act_tensor[:,class_idx].numpy(), xlabel='Activation', ylabel='Numbers', 
              title=f'{class_name}',
              save_path=f'{working_path}/img_distribution/{class_name}.jpg')
    
    
dataset_num = acc_info['video_num'].sum()

acc_sum1 = ((acc_info['video_num']*acc_info['acc1']).astype('int64')).sum()/dataset_num
acc_sum5 = ((acc_info['video_num']*acc_info['acc5']).astype('int64')).sum()/dataset_num

acc_info.to_csv(f'{working_path}/acc_info.csv', index=False)



#%% random 

for idx,class_name in enumerate(os.listdir(input_path)[:2]):
    video_path = os.path.join(input_path, class_name)
    video_name = np.load(video_path)























