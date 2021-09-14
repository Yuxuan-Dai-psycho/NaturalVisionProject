import torch
import numpy as np
import pickle as pkl
import torch.nn.functional as F

from torch import nn
from os.path import join as pjoin
from model_utils import CustomizedLinear

# define network
n_class = 10 # the classification class num
class VisualNet(nn.Module):
    def __init__(self, p, selected_voxel=None):
        super(VisualNet, self).__init__()
        # load mask info
        # define path
        main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
        out_path = pjoin(main_path, 'imagenet_decoding')
        # load brain structure info
        with open(pjoin(out_path, 'dnn_struct_info.pkl'), 'rb') as f:
            dnn_struct_info = pkl.load(f)
        # load mask
        voxel2roi_mask = dnn_struct_info['voxel2roi_mask']
        roi2network_mask = dnn_struct_info['roi2network_mask']
        del dnn_struct_info
        
        # voxel selection before network
        if selected_voxel is not None:
            # for voxel2roi mask
            voxel2roi_mask = voxel2roi_mask[selected_voxel, :]
            selected_roi = np.sum(voxel2roi_mask, axis=0) != 0
            voxel2roi_mask = torch.tensor(voxel2roi_mask[:, selected_roi])
            # for voxel2roi mask
            roi2network_mask = roi2network_mask[selected_roi, :]
            selected_net = np.sum(roi2network_mask, axis=0) != 0
            roi2network_mask = torch.tensor(roi2network_mask[:, selected_net])
            print(f'Select {selected_voxel.shape[0]} Voxels; ' 
                  f'{selected_roi.sum()} ROIs; {selected_net.sum()} Networks')
            
        # define layer
        self.voxel2roi = CustomizedLinear(voxel2roi_mask)
        self.roi2network = CustomizedLinear(roi2network_mask)
        self.output = nn.Linear(roi2network_mask.size()[1], n_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.voxel2roi(x)
        x = self.roi2network(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)  
        return x


class VisualNet_fully_connected(nn.Module):
    def __init__(self, p):
        super(VisualNet_fully_connected, self).__init__()            
        # define layer
        self.hidden1 = nn.Linear(2757, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.hidden2 = nn.Linear(64, 4)
        self.bn2 = nn.BatchNorm1d(num_features=4)
        self.output = nn.Linear(4, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        # x = self.bn2(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)  
        return x


class VisualNet_simple(nn.Module):
    def __init__(self, p, n_hidden):
        super(VisualNet_simple, self).__init__()            
        # define layer
        self.hidden1 = nn.Linear(2757, n_hidden)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden)
        self.output = nn.Linear(n_hidden, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.hidden1(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        # x = self.bn2(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)  
        return x



