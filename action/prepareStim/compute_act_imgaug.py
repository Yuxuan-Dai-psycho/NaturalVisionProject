import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from network.srtg_resnet import srtg_r2plus1d_50, srtg_r3d_101
# from network.MTnet import MTNet
# from torchvision import transforms
from data.video_sampler import RandomSampling
import imgaug.augmenters as iaa
import data.video_transforms as transforms

#%% Define params
# define path
dataset_path = '/nfs/e3/VideoDatabase/HACS/training'
working_path = '/nfs/s2/userhome/zhouming/workingdir/Video/HACS/train_model'
output_path  = f'{working_path}/out'
# test_video = 'v_Playing polo_id_1_MVlR0hhEc_start_0.5_label_1.mp4'

#%% prepare model

model = srtg_r2plus1d_50(num_classes=200)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    info = torch.load(f'{working_path}/models/srtg_r2plus1d_50_best.pth')
    model.load_state_dict({k.replace('module.',''):v for k,v in info['state_dict'].items()}, 
                          strict = False)
else:
    info = torch.load(f'{working_path}/models/srtg_r2plus1d_50_best.pth', map_location='cpu')
    model.load_state_dict({k.replace('module.',''):v for k,v in info['state_dict'].items()}, 
                          strict = False)
# info['state_dict']['module.fc.weight'].shape

# set transform
clip_size = 256
sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
sometimes_seq = lambda aug: iaa.Sometimes(0.8, aug)
video_transform = transforms.Compose(
                        transforms = iaa.Sequential([
                                iaa.Resize({"shorter-side": 384, "longer-side":"keep-aspect-ratio"}),
                                iaa.CropToFixedSize(width=384, height=384, position='center'),
                                iaa.CropToFixedSize(width=clip_size, height=clip_size, position='uniform'),
                                sometimes_seq(iaa.Sequential([
                                    sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                    sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                    sometimes_aug(iaa.AverageBlur(k=(1,2))),
                                    sometimes_aug(iaa.Multiply((0.8, 1.2))),
                                    sometimes_aug(iaa.GammaContrast((0.85,1.15),per_channel=True)),
                                    sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                    sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                    sometimes_aug(
                                        iaa.OneOf([
                                            iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                            iaa.Rotate(rotate=(-10,10)),
                                        ])
                                    )
                                ])),
                                iaa.Fliplr(0.5)]),
                        normalise = [[0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]]
                        )

#%%  prepare dataset
# set params
frame_num = 8 # random sub-sampling of 8 frames
frame_size = (clip_size, clip_size)
class_sample = 480 # num of class using in exp
test_num = 5
# batch_size = 32
# set video path
data = pd.read_csv(f'{working_path}/duration.csv')
data = data.sort_values(ascending=True, by='class')
qualified = data[(data['duration'] < 2.05) & (data['duration'] > 1.95) & (data['subset']=='training')]
qualified['label'] = qualified['label'].astype('int64')
del data
time_start = time.time()

for class_idx,class_name in enumerate(qualified['class'].unique()[:2]):
    if qualified[(qualified['class']==class_name) & (qualified['label']==1)].shape[0] < class_sample:
        video_class = qualified[(qualified['class']==class_name)]
    else:
        video_class = qualified[(qualified['class']==class_name) & (qualified['label']==1)]

    video_paths = ['/'.join([dataset_path, class_name, x]) for x in video_class['video'].tolist()]
    if test_num != None:
        video_paths = video_paths[:test_num]
    print(f'Start creating tensor for class {class_name} in {class_idx+1}/200--------')
    # prepare data loader
    data = torch.zeros(len(video_paths), 3, frame_num, *frame_size)
    for vid_idx,vid_file in enumerate(video_paths):
        vid_cap = cv2.VideoCapture(vid_file)
        frame_sum = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # check invalid video and pass it
        if frame_sum == 0:
            print(f'Find a error video in {vid_file}')
            continue
        random_sampler = RandomSampling(num=frame_num, interval=2)
        random_idx = random_sampler.sampling(range_max=frame_sum)
        # initialize tensor for one video
        frames = np.zeros((frame_num, height, width, 3), dtype=np.uint8)
        for loop_idx,frame_idx in enumerate(random_idx):
            # get frame
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)
            _, frame = vid_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[loop_idx] = frame
        # augmentation and transformation
        data[vid_idx] = video_transform(frames, (3, *frame_size))
        time_spent = time.time() - time_start
        print('Finish preparing videos: %d/%d in class %d/200. Time spent %.2fs' \
              %(vid_idx+1, len(video_paths), class_idx+1, time_spent))
    # dataset = TensorDataset(data)
    # data_loader = DataLoader(dataset, batch_size)
    # save data
    # torch.save(dataset, f'{output_path}/input/{class_name}_input_tensor.pt')
    # np.save(f'{output_path}/input/{class_name}_input_path.npy', np.asarray(video_paths))
    
    #% compute activation
    print(f'Start computing activation for class {class_name} in {class_idx+1}/200--------')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc_tensor = torch.zeros((len(video_paths), 200))
    with torch.no_grad():
        # video_test = torch.rand(1,3,8,256,256) # 3 means channel; 8 means frame num; 256 means frame height & width
        model.eval()
        # model.cuda(1)#to(device)
        for vid_idx in range(len(video_paths)):
            input_tensor = data[vid_idx].unsqueeze(0)#.cuda(1)#to(device)
            output = model(input_tensor)
            print(output)
            print(torch.max(output, dim=1).indices)
            fc_tensor[vid_idx] = output
            time_spent = time.time() - time_start
            print('Finish computing activation: %d/%d in class %d/200. Time spent %.2fs' \
                  %(vid_idx+1, len(video_paths), class_idx+1, time_spent))
    # save data
    # torch.save(fc_tensor, f'{output_path}/fc/{class_name}_fc_tensor.pt')

