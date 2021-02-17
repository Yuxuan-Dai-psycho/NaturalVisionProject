import os
import cv2
import torch
import numpy as np
from os.path import join as pjoin
from dnnbrain.dnn.base import VideoClipSet
from network.srtg_resnet import srtg_r2plus1d_50

#%% Define params
# define path
dataset_path = 'D:/fMRI/action/stimulus/video'
working_path = 'D:/deepLearning/train_model'
test_video = 'v_Playing polo_id_1_MVlR0hhEc_start_0.5_label_1.mp4'

#%% prepare video
test_class = 'Archery'
video_names = os.listdir(pjoin(dataset_path, test_class))
video_paths = ['/'.join([dataset_path, test_class, x]) for x in video_names]

# random sub-sampling of 16 frames
video = VideoClipSet(video_paths)
print('Video Number: %d'%(video.__len__()))

#%%
model = srtg_r2plus1d_50(num_classes=200)
info = torch.load(f'{working_path}/srtg_r2plus1d_50_best.pth', map_location='cpu')
model.load_state_dict(info['state_dict'], False)

#%% 
with torch.no_grad():
    # video_test = torch.rand(1,3,8,256,256) # 3 means channel; 8 means frame num; 256 means frame height & width
    model.eval()
    for idx in range(video.__len__()):
        target_tensor = (video.__getitem__(idx)[0]).unsqueeze(0)
        output = model(target_tensor)
        _, predicted = torch.max(output, 1)
        print(predicted)
    

